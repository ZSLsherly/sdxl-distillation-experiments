"""Batch indexing, per-sample randomness, and differentiable batched VAE tiles."""
from dataclasses import dataclass
import random

import torch
from torch.utils.checkpoint import checkpoint


@dataclass(frozen=True)
class BatchPlan:
    global_batch: int
    world_size: int
    micro_batch: int

    def __post_init__(self):
        if min(self.global_batch, self.world_size, self.micro_batch) < 1:
            raise ValueError("Batch sizes and world_size must be positive")
        if self.global_batch % (self.world_size * self.micro_batch):
            raise ValueError("global_batch must be divisible by world_size * micro_batch")

    @property
    def accumulation(self):
        return self.global_batch // (self.world_size * self.micro_batch)

    def positions(self, update, micro, rank):
        # Changing micro_batch preserves each rank's ordered sample stream.
        return [update * self.global_batch + (micro * self.micro_batch + j) * self.world_size + rank
                for j in range(self.micro_batch)]

    def rank_positions(self, update, rank):
        return [update * self.global_batch + j * self.world_size + rank
                for j in range(self.global_batch // self.world_size)]


def conditions(data, indices, device):
    ids = torch.tensor(indices, dtype=torch.long)
    batch = len(indices)
    result = {}
    for key in ("prompt_embeds", "negative_prompt_embeds", "pooled_prompt_embeds",
                "negative_pooled_prompt_embeds", "time_ids"):
        source = data[key]
        value = source.expand(batch, *source.shape[1:]) if source.shape[0] == 1 else source.index_select(0, ids)
        result[key] = value.to(device)
    return result


def random_inputs(positions, phase, device, seed=20261017, shape=(4, 128, 128)):
    """Sampling is independent of packing, rank timing, and GAN/DMD branch choice.

    The small per-image RNG loop never calls a neural network. All model calls
    receive the concatenated batch.
    """
    if phase not in ("fake", "student"):
        raise ValueError(phase)
    values = {k: [] for k in ("noise", "t", "eps", "pcm_t", "pcm_eps")}
    crops, flips = [], []
    offset = 100000 if phase == "fake" else 200000
    for pos in positions:
        generator = torch.Generator(device=device).manual_seed(seed + offset + pos)
        randn = lambda: torch.randn(shape, device=device, generator=generator)
        values["noise"].append(randn())
        lo, hi = (0, 1000) if phase == "fake" else (20, 920)
        values["t"].append(torch.randint(lo, hi, (), device=device, generator=generator))
        values["eps"].append(randn())
        if phase == "student":
            values["pcm_t"].append(torch.randint(940, 1000, (), device=device, generator=generator))
            values["pcm_eps"].append(randn())
        spatial = random.Random(seed + offset + pos + 900000)
        crops.append((spatial.randrange(shape[-2] - 48 + 1), spatial.randrange(shape[-1] - 48 + 1)))
        flips.append(spatial.random() < 0.5)
    result = {k: torch.stack(v) for k, v in values.items() if v}
    result.update(crops=crops, flips=torch.tensor(flips, device=device).reshape(-1, 1, 1, 1))
    return result


def decode_tiles(vae, latents, crops, grad=False, chunk_size=None):
    """Pack independent 48x48 latent crops, then make ONE batched VAE call.

    GroupNorm makes tile decoding approximate relative to full-frame decoding;
    this preserves v4's training crop definition, including its 64px halo.
    """
    if len(crops) != latents.shape[0]:
        raise ValueError("One crop is required per image")
    tiles = torch.stack([latents[i, :, y:y + 48, x:x + 48] for i, (y, x) in enumerate(crops)])
    tiles = tiles / vae.config.scaling_factor

    def decode(z):
        return vae.decode(z.float(), return_dict=False)[0][:, :, 64:320, 64:320]

    if chunk_size is None:
        chunk_size = len(tiles)
    outputs = [checkpoint(decode, chunk, use_reentrant=False) if grad else decode(chunk)
               for chunk in tiles.split(chunk_size)]
    return outputs[0] if len(outputs) == 1 else torch.cat(outputs)


def flip_images(rgb, mask):
    return torch.where(mask, rgb.flip(-1), rgb)


def slice_batch(values, start, end):
    """Slice tensors and per-image crop lists with the same sample boundaries."""
    return {key: value[start:end] for key, value in values.items()}


@torch.no_grad()
def rollout_chunks(rollout, condition, noise, chunk_size):
    """Generate one local effective batch while student weights are unchanged.

    All ranks have the same local sample count, including any final short
    chunk, so their FSDP forwards occur in the same order. This cache is only
    valid until the next student optimizer step and never includes a graph.
    """
    if chunk_size < 1:
        raise ValueError("rollout batch must be positive")
    chunks = []
    for start in range(0, noise.shape[0], chunk_size):
        end = start + chunk_size
        chunks.append(rollout(slice_batch(condition, start, end), noise[start:end]).detach())
    return torch.cat(chunks)


def cfg_prediction(predict, model, x, t, condition, scale, batched=False):
    if batched:
        paired = {key: torch.cat([value, value]) for key, value in condition.items()}
        paired["prompt_embeds"] = torch.cat([condition["negative_prompt_embeds"], condition["prompt_embeds"]])
        paired["pooled_prompt_embeds"] = torch.cat([condition["negative_pooled_prompt_embeds"], condition["pooled_prompt_embeds"]])
        uncond, cond = predict(model, torch.cat([x, x]), torch.cat([t, t]), paired).chunk(2)
    else:
        uncond = predict(model, x, t, condition, True)
        cond = predict(model, x, t, condition)
    return uncond + scale * (cond - uncond)


def r1_penalty(scores, real_rgb):
    # Sum before differentiating, mean over examples afterwards. Taking a mean
    # of scores first would shrink R1 by batch_size**2.
    derivative = torch.autograd.grad(scores.sum(), real_rgb, create_graph=True)[0]
    return derivative.square().flatten(1).sum(1).mean()
