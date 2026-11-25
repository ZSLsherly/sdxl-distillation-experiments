import json
from pathlib import Path
from unittest import TestCase, main
import torch
import boundary as bd
from terminal_ablation import student_dm_term, gradient_probe


class AblationTests(TestCase):
    def test_loss_and_probe(self):
        p = torch.tensor([1., 2.], dtype=torch.float64, requires_grad=True)
        d = lambda: (p * torch.tensor([-1., 3.])).sum()
        c = lambda: (p * torch.tensor([2., -1.])).sum()
        def clear():
            p.grad = None
        def norm():
            return p.grad.norm().item()
        rng = torch.get_rng_state().clone()
        for w in (1, 2):
            stats = gradient_probe(d, c, clear, norm, w, 1., 'cpu', 37)
            self.assertIsNone(p.grad)
            (c() + student_dm_term(d(), 4, False, w)).backward()
            expected = torch.tensor([2.-w, -1.+3*w], dtype=p.dtype)
            torch.testing.assert_close(p.grad, expected)
            self.assertAlmostEqual(stats['joint_grad_norm'], expected.norm().item())
            self.assertAlmostEqual(stats['cosine'], -5 / (50**0.5))
            self.assertAlmostEqual(stats['weighted_dm_over_pcm'], w * 2**0.5)
            clear()
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        self.assertEqual(student_dm_term(d(), 2, True, 2).item(), 0.5*d().item())

    def test_real_boundary_surrogate_gradient_scaling(self):
        from objectives import VPSchedule
        schedule = VPSchedule(torch.linspace(.998, .01, 1000))
        clean = torch.tensor([[[[.3, -.1]]]], requires_grad=True)
        t = torch.tensor([60])
        eps = torch.tensor([[[[.25, -.4]]]])
        f = torch.zeros_like(clean)
        teacher = torch.full_like(clean, .2)
        grads = []
        for w in (1., 2.):
            xt = bd.renoise(schedule, clean.detach(), -1, t, eps)
            dm, _ = bd.dm_surrogate(schedule, clean, -1, t, xt, f, teacher)
            term = student_dm_term(dm, 4, False, w)
            grads.append(torch.autograd.grad(term, clean)[0])
        self.assertGreater(grads[0].norm().item(), 0)
        torch.testing.assert_close(grads[1], 2 * grads[0])

    def test_matched_recipe_and_fake_schedule(self):
        cfgdir = Path(__file__).parent / 'configs'
        x1, x2 = [json.loads((cfgdir / f'dm4_x{w}.json').read_text()) for w in (1,2)]
        self.assertEqual(x2, dict(x1, terminal_dm_weight=2))
        self.assertTrue(x2['terminal_only_dm'] and x2['pair_fake_stages'])
        for g in range(160):
            stages = [bd.critic_stage(2*g+i, True, x2['seed'], x2['fake_updates']) for i in range(2)]
            self.assertEqual(stages, [bd.middle_stage(g, x2['seed']), 4])


if __name__ == '__main__':
    main()
