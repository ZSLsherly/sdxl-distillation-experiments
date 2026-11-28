"""CPU contract check for the exact functions used by this deployment."""
import torch
from boundary import local_rollout
from objectives import VPSchedule, rollout_score_and_final, dmd_surrogate


def main():
    torch.manual_seed(12)
    knots = [999, 749, 499, 249, -1]
    sched = VPSchedule(torch.linspace(.999, .015, 1000))
    noise = torch.randn(2, 4, 3, 3)
    for stage in [2, 3]:
        weights = [torch.tensor(.1*k, requires_grad=True) for k in range(1, 5)]
        def pred(x, t, cond):
            return x * weights[knots.index(t[0].item())]
        x = noise
        refs = {}
        with torch.no_grad():
            for k in range(1, 5):
                t = torch.full((2,), knots[k-1])
                eps = pred(x, t, None)
                clean = sched.clean(x, eps, t)
                x = sched.jump(x, eps, t, knots[k])
                refs[k] = clean
                if k == stage:
                    assert not torch.allclose(clean, x)
        fake_input, terminal = rollout_score_and_final(pred, sched, None, noise, knots, stage)
        assert torch.equal(fake_input, refs[stage]) and torch.equal(terminal, refs[4])
        assert not fake_input.requires_grad
        observed = []
        for k, clean in local_rollout(pred, sched, None, noise, knots, stage):
            observed.append(k)
            assert torch.equal(clean, refs[k])
            grads = torch.autograd.grad(clean.sum(), weights, allow_unused=True)
            assert [i+1 for i,g in enumerate(grads) if g is not None] == [k]
        assert observed == [stage, 4]
    clean = noise.detach().requires_grad_()
    t = torch.tensor([20, 979])
    eps = torch.randn_like(clean)
    with torch.no_grad():
        xt = sched.noise(clean, eps, t)
        torch.testing.assert_close(sched.clean(xt, eps, t), clean, atol=2e-6, rtol=2e-6)
        f = sched.clean(xt, .75*eps, t)
        r = sched.clean(xt, .8*eps, t)
    loss, direction = dmd_surrogate(clean, f, r)
    torch.testing.assert_close(torch.autograd.grad(loss*.5, clean)[0], direction/clean.numel())
    print("PASS clean output, paired fake inputs, local gradients, global renoise and DM gradient")


if __name__ == '__main__':
    main()
