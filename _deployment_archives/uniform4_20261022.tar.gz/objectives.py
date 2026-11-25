"""The existing SDXL VP/DDIM and DMD objective, with per-image normalization."""
import torch


class VPSchedule:
    def __init__(self, alphas_cumprod):
        self.alpha = alphas_cumprod.float().sqrt()
        self.sigma = (1 - alphas_cumprod.float()).sqrt()

    def coefficients(self, t, x):
        t = torch.as_tensor(t, device=x.device, dtype=torch.long).reshape(-1)
        a = self.alpha.to(x.device)[t.clamp_min(0)]
        s = self.sigma.to(x.device)[t.clamp_min(0)]
        a = torch.where(t < 0, torch.ones_like(a), a)
        s = torch.where(t < 0, torch.zeros_like(s), s)
        shape = (-1,) + (1,) * (x.ndim - 1)
        return a.reshape(shape), s.reshape(shape)

    def noise(self, clean, eps, t):
        a, s = self.coefficients(t, clean)
        return a * clean + s * eps

    def clean(self, x, eps, t):
        a, s = self.coefficients(t, x)
        return (x - s * eps) / a

    def jump(self, x, eps, t, next_t):
        a, s = self.coefficients(next_t, x)
        return a * self.clean(x, eps, t) + s * eps


def dmd_surrogate(generated, fake_clean, teacher_clean):
    with torch.no_grad():
        scale = (generated - teacher_clean).abs().flatten(1).mean(1).clamp_min(1e-5)
        scale = scale.reshape(-1, *([1] * (generated.ndim - 1)))
        direction = (fake_clean - teacher_clean) / scale
        target = (generated - direction).detach()
    return (generated - target).square().flatten(1).mean(1).mean(), direction


def discriminator_loss(real, fake):
    return torch.relu(1 - real).mean() + torch.relu(1 + fake).mean()
