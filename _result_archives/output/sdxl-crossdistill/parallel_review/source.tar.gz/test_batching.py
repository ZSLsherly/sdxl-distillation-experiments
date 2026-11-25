"""CPU numerical checks; no model downloads, files, or GPUs are required."""
import copy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

from batching import (BatchPlan, cfg_prediction, conditions, decode_tiles, flip_images,
                      r1_penalty, random_inputs, rollout_chunks, slice_batch)
from objectives import VPSchedule, discriminator_loss, dmd_surrogate


class ToyVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(scaling_factor=.5)
        self.calls = 0

    def decode(self, x, return_dict=False):
        self.calls += 1
        return (F.interpolate(F.group_norm(x[:, :3], 1), scale_factor=8, mode="nearest"),)


def example_data():
    return {"prompt_embeds": torch.randn(7, 5, 8), "pooled_prompt_embeds": torch.randn(7, 4),
            "negative_prompt_embeds": torch.zeros(1, 5, 8), "negative_pooled_prompt_embeds": torch.zeros(1, 4),
            "time_ids": torch.tensor([[1024, 1024, 0, 0, 1024, 1024]], dtype=torch.float32)}


class BatchTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20261017)
        torch.set_num_threads(2)

    def test_batch64_covers_exactly_64_distinct_positions(self):
        for size in (1, 2, 4, 8, 16, 32):
            plan = BatchPlan(64, 2, size)
            positions = [p for rank in range(2) for micro in range(plan.accumulation)
                         for p in plan.positions(3, micro, rank)]
            self.assertEqual(sorted(positions), list(range(192, 256)))

    def test_batch_packing_preserves_rank_sample_order(self):
        for rank in range(2):
            expected = None
            for size in (1, 2, 4, 8):
                plan = BatchPlan(64, 2, size)
                stream = [p for m in range(plan.accumulation) for p in plan.positions(1, m, rank)]
                if expected is None:
                    expected = stream
                self.assertEqual(stream, expected)
                self.assertEqual(stream, plan.rank_positions(1, rank))

    def test_invalid_batch_sizes(self):
        for values in ((64, 2, 3), (0, 2, 1), (64, 0, 1)):
            with self.assertRaises(ValueError):
                BatchPlan(*values)

    def test_text_and_time_conditions_expand_to_actual_batch(self):
        data = example_data()
        batch = conditions(data, [5, 1, 3], "cpu")
        for key in batch:
            self.assertEqual(batch[key].shape[0], 3)
        torch.testing.assert_close(batch["prompt_embeds"], data["prompt_embeds"][[5, 1, 3]])
        torch.testing.assert_close(batch["time_ids"], data["time_ids"].expand(3, -1))
        # Also support per-image negative conditioning, rather than silently
        # taking the first negative prompt for all examples.
        data["negative_prompt_embeds"] = torch.randn(7, 5, 8)
        torch.testing.assert_close(conditions(data, [6, 0], "cpu")["negative_prompt_embeds"],
                                   data["negative_prompt_embeds"][[6, 0]])

    def test_noise_timesteps_crops_do_not_depend_on_microbatch(self):
        for phase in ("student", "fake"):
            together = random_inputs([10, 12, 14, 16], phase, "cpu", shape=(4, 64, 64))
            chunks = [random_inputs([i], phase, "cpu", shape=(4, 64, 64)) for i in (10, 12, 14, 16)]
            for key, value in together.items():
                if key == "crops":
                    self.assertEqual(value, [c["crops"][0] for c in chunks])
                else:
                    torch.testing.assert_close(value, torch.cat([c[key] for c in chunks]), rtol=0, atol=0)

    def test_batched_timestep_transport_matches_per_image(self):
        schedule = VPSchedule(torch.linspace(.999, .004, 1000))
        clean, eps = torch.randn(4, 4, 8, 8), torch.randn(4, 4, 8, 8)
        t, nxt = torch.tensor([999, 950, 600, 200]), torch.tensor([919, 919, 499, -1])
        xt = schedule.noise(clean, eps, t)
        jumped = schedule.jump(xt, eps, t, nxt)
        oracle = schedule.noise(clean, eps, nxt)
        torch.testing.assert_close(jumped, oracle, rtol=1e-5, atol=3e-6)
        per_image = torch.cat([schedule.jump(xt[i:i + 1], eps[i:i + 1], t[i], nxt[i]) for i in range(4)])
        torch.testing.assert_close(jumped, per_image)

    def test_batched_crop_uses_one_vae_call_and_preserves_pixel_values(self):
        x = torch.randn(3, 4, 64, 64)
        crops = [(0, 0), (7, 13), (16, 16)]
        vae = ToyVAE()
        actual = decode_tiles(vae, x, crops)
        self.assertEqual(vae.calls, 1)
        expected = torch.cat([decode_tiles(ToyVAE(), x[i:i + 1], [crop]) for i, crop in enumerate(crops)])
        self.assertEqual(actual.shape, (3, 3, 256, 256))
        torch.testing.assert_close(actual, expected)

    def test_crop_and_checkpoint_keep_per_sample_gradients(self):
        x = torch.randn(2, 4, 64, 64, requires_grad=True)
        crops = [(0, 0), (16, 16)]
        y = decode_tiles(ToyVAE(), x, crops, grad=True)
        target = torch.randn_like(y)
        (y - target).square().mean().backward()
        reference = x.detach().clone().requires_grad_(True)
        for i, crop in enumerate(crops):
            z = decode_tiles(ToyVAE(), reference[i:i + 1], [crop])
            ((z - target[i:i + 1]).square().mean() / 2).backward()
        torch.testing.assert_close(x.grad, reference.grad, rtol=1e-4, atol=1e-7)
        self.assertGreater(x.grad.abs().sum().item(), 0)
        self.assertEqual(x.grad[0, :, 48:, :].abs().sum().item(), 0)
        self.assertEqual(x.grad[1, :, :16, :].abs().sum().item(), 0)

    def test_per_sample_flip_does_not_mix_images(self):
        x = torch.arange(24).reshape(2, 1, 3, 4)
        result = flip_images(x, torch.tensor([True, False]).reshape(2, 1, 1, 1))
        torch.testing.assert_close(result[0], x[0].flip(-1))
        torch.testing.assert_close(result[1], x[1])

    def test_rollout_cache_has_correct_order_tail_and_no_graph(self):
        data = example_data()
        condition = conditions(data, [0, 1, 2, 3, 4, 5, 6, 0, 1, 2], "cpu")
        noise = torch.randn(10, 4, 8, 8, requires_grad=True)
        weight = nn.Parameter(torch.tensor(.8))
        seen = []

        def generate(c, x):
            seen.append(x.shape[0])
            offset = c["pooled_prompt_embeds"].mean(1).reshape(-1, 1, 1, 1)
            return weight * (x + offset).tanh()

        cached = rollout_chunks(generate, condition, noise, 4)
        self.assertEqual(seen, [4, 4, 2])
        seen.clear()
        reference = rollout_chunks(generate, condition, noise, 2)
        self.assertEqual(seen, [2] * 5)
        torch.testing.assert_close(cached, reference)
        self.assertFalse(cached.requires_grad)
        self.assertIsNone(weight.grad)
        self.assertIsNone(noise.grad)

    def test_cached_generation_preserves_fake_D_and_R1_gradients(self):
        condition = conditions(example_data(), [0, 1, 2, 3, 4, 5, 6, 0], "cpu")
        noise, real = torch.randn(8, 4, 8, 8), torch.randn(8, 4, 8, 8)
        target = torch.randn_like(noise)

        def generate(c, x):
            offset = c["pooled_prompt_embeds"].mean(1).reshape(-1, 1, 1, 1)
            return (x + offset).tanh()

        cached = rollout_chunks(generate, condition, noise, 8)
        fake = nn.Conv2d(4, 4, 1)
        discriminator = nn.Conv2d(4, 1, 1)
        reference_fake, reference_D = copy.deepcopy(fake), copy.deepcopy(discriminator)

        def backward(f, d, samples, j):
            real_input = real[j:j + 2].clone().requires_grad_(True)
            logits = d(real_input).flatten(1).mean(1)
            fake_logits = d(samples).flatten(1).mean(1)
            loss = (f(samples) - target[j:j + 2]).square().mean()
            loss = loss + discriminator_loss(logits, fake_logits) + .8 * r1_penalty(logits, real_input)
            (loss / 4).backward()

        for j in range(0, 8, 2):
            backward(fake, discriminator, cached[j:j + 2], j)
            with torch.no_grad():
                inline = generate(slice_batch(condition, j, j + 2), noise[j:j + 2])
            backward(reference_fake, reference_D, inline, j)
        for a, b in zip(list(fake.parameters()) + list(discriminator.parameters()),
                        list(reference_fake.parameters()) + list(reference_D.parameters())):
            torch.testing.assert_close(a.grad, b.grad)

    def test_batched_teacher_CFG_matches_two_calls(self):
        condition = conditions(example_data(), [2, 5, 1], "cpu")
        x, t = torch.randn(3, 4, 8, 8), torch.tensor([999, 499, 199])
        sizes = []

        def predict(model, x, t, c, negative=False):
            sizes.append(x.shape[0])
            emb = c["negative_prompt_embeds" if negative else "prompt_embeds"]
            pooled = c["negative_pooled_prompt_embeds" if negative else "pooled_prompt_embeds"]
            offset = emb.mean((1, 2)) + pooled.mean(1) + t / 1000 + c["time_ids"].mean(1) / 10000
            return (x + offset.reshape(-1, 1, 1, 1)).tanh()

        separate = cfg_prediction(predict, None, x, t, condition, 8, False)
        self.assertEqual(sizes, [3, 3])
        sizes.clear()
        batched = cfg_prediction(predict, None, x, t, condition, 8, True)
        self.assertEqual(sizes, [6])
        torch.testing.assert_close(batched, separate)

    def test_dmd_batch_mean_matches_accumulated_per_image_gradients(self):
        x = torch.randn(4, 4, 8, 8)
        fake, teacher = torch.randn_like(x), torch.randn_like(x)
        weight = nn.Parameter(torch.tensor(.8))
        loss, _ = dmd_surrogate(weight * x, fake, teacher)
        loss.backward()
        expected = weight.grad.clone()
        weight.grad = None
        for i in range(4):
            loss, _ = dmd_surrogate(weight * x[i:i + 1], fake[i:i + 1], teacher[i:i + 1])
            (loss / 4).backward()
        torch.testing.assert_close(weight.grad, expected)

    def test_r1_strength_is_independent_of_physical_batch(self):
        x = torch.randn(4, 3, 4, 4, requires_grad=True)
        weight = nn.Parameter(torch.tensor(.7))
        scores = (weight * x.square()).flatten(1).mean(1)
        r1 = r1_penalty(scores, x)
        r1.backward()
        expected_value, expected_grad = r1.detach(), weight.grad.clone()
        weight.grad = None
        values = []
        for i in range(4):
            z = x.detach()[i:i + 1].requires_grad_(True)
            value = r1_penalty((weight * z.square()).flatten(1).mean(1), z)
            (value / 4).backward()
            values.append(value.detach())
        torch.testing.assert_close(torch.stack(values).mean(), expected_value)
        torch.testing.assert_close(weight.grad, expected_grad)

    def test_accumulation_and_rank_average_equal_global_batch_gradient(self):
        x, target = torch.randn(16, 3), torch.randn(16, 2)
        original = nn.Linear(3, 2)
        (original(x) - target).square().mean().backward()
        for size in (1, 2, 4, 8):
            grads = []
            plan = BatchPlan(16, 2, size)
            for rank in range(2):
                model = copy.deepcopy(original)
                model.zero_grad(set_to_none=True)
                for micro in range(plan.accumulation):
                    positions = plan.positions(0, micro, rank)
                    loss = (model(x[positions]) - target[positions]).square().mean()
                    (loss / plan.accumulation).backward()
                grads.append([p.grad.clone() for p in model.parameters()])
            for i, param in enumerate(original.parameters()):
                torch.testing.assert_close((grads[0][i] + grads[1][i]) / 2, param.grad, rtol=1e-5, atol=1e-7)

    def test_actual_diffusers_sdxl_conditioning_accepts_multi_image_batch(self):
        from diffusers import UNet2DConditionModel
        model = UNet2DConditionModel(
            sample_size=8, in_channels=4, out_channels=4, layers_per_block=1,
            block_out_channels=(8, 16), down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"), cross_attention_dim=8,
            attention_head_dim=2, norm_num_groups=4, addition_embed_type="text_time",
            addition_time_embed_dim=2, projection_class_embeddings_input_dim=16).eval()
        reference = copy.deepcopy(model)
        data = example_data()
        x = torch.randn(3, 4, 8, 8)
        times = torch.tensor([999, 499, 199])
        c = conditions(data, [1, 3, 5], "cpu")

        def forward(m, z, t, condition):
            return m(z, t, encoder_hidden_states=condition["prompt_embeds"],
                     added_cond_kwargs={"text_embeds": condition["pooled_prompt_embeds"],
                                        "time_ids": condition["time_ids"]}, return_dict=False)[0]

        y = forward(model, x, times, c)
        y.square().mean().backward()
        ys = []
        for i in range(3):
            yi = forward(reference, x[i:i + 1], times[i:i + 1], {k: v[i:i + 1] for k, v in c.items()})
            ys.append(yi.detach())
            (yi.square().mean() / 3).backward()
        torch.testing.assert_close(y.detach(), torch.cat(ys), rtol=1e-4, atol=2e-6)
        for a, b in zip(model.parameters(), reference.parameters()):
            if a.grad is not None:
                torch.testing.assert_close(a.grad, b.grad, rtol=5e-4, atol=2e-6)

    def make_small_pixel_discriminator(self):
        from transformers import Dinov2Config, Dinov2Model
        from pixel import PixelDiscriminator
        config = Dinov2Config(hidden_size=24, num_hidden_layers=12,
                             num_attention_heads=3, image_size=224, patch_size=14,
                             mlp_ratio=2)
        config._attn_implementation = "eager"
        backbone = Dinov2Model(config)
        with patch("pixel.Dinov2Model.from_pretrained", return_value=backbone):
            model = PixelDiscriminator("unused-no-download")
        return model

    def test_pixel_gan_batch_keeps_input_gradients_and_frozen_backbone(self):
        model = self.make_small_pixel_discriminator()
        # Initialize the spectral-normalization buffers before eval mode.
        model.set_heads(True)
        with torch.no_grad():
            model(torch.randn(2, 3, 256, 256), torch.randn(2, 1280))
        model.set_heads(False)
        image = torch.randn(2, 3, 256, 256, requires_grad=True)
        result = model(image, torch.randn(2, 1280), return_parts=True)
        self.assertEqual(result["total"].shape, (2,))
        self.assertEqual(result["dino_heads"].shape, (4, 2))
        (-result["total"].mean()).backward()
        self.assertTrue(torch.isfinite(image.grad).all())
        self.assertTrue((image.grad.abs().flatten(1).sum(1) > 0).all())
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_pixel_gan_batched_r1_supports_second_backward(self):
        model = self.make_small_pixel_discriminator()
        model.set_heads(True)
        image = torch.randn(2, 3, 256, 256, requires_grad=True)
        scores = model(image, torch.randn(2, 1280))
        penalty = r1_penalty(scores, image)
        penalty.backward()
        self.assertTrue(torch.isfinite(penalty))
        self.assertTrue(all(p.grad is None for p in model.backbone.parameters()))
        grads = [p.grad for p in model.trainable() if p.grad is not None]
        self.assertTrue(grads)
        self.assertTrue(all(torch.isfinite(g).all() for g in grads))


if __name__ == "__main__":
    unittest.main(verbosity=2)
