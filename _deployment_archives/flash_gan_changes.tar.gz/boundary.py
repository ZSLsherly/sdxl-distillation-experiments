"""VP boundary distribution matching; no model loading or training framework.

This is a conditional-score adaptation, NOT TDM's importance-weighted target.
The terminal clean boundary is -1 (alpha=1, sigma=0), not DDIM index 0.
"""
import random
import torch


def phase_ranges(knots, margin=10, noise_max=None):
    if len(knots) != 5 or knots[-1] != -1 or margin < 1:
        raise ValueError('Expected four descending inference knots plus clean -1')
    if any(a <= b for a, b in zip(knots, knots[1:])):
        raise ValueError('Inference knots must descend')
    ranges = {k: (knots[k] + margin if k < 4 else 20, knots[k-1] - margin)
              for k in (2, 3, 4)}
    if noise_max is not None:
        if not 1 <= noise_max <= 999:
            raise ValueError("noise_max must be in [1,999]")
        ranges = {k: (lo, noise_max) for k, (lo, _) in ranges.items()}
    if any(lo > hi for lo, hi in ranges.values()):
        raise ValueError('No room for boundary re-noising with this margin')
    return ranges


def middle_stage(update, seed):
    # Same choice on every rank, reproducible without depending on resume RNG.
    return random.Random(seed + 310000 + update).choice((2, 3))


def critic_stage(update, paired=False, seed=20261017):
    if paired:
        return middle_stage(update // 2, seed) if update % 2 == 0 else 4
    return 2 + update % 3


def coefficients(schedule, boundary, time, sample):
    alpha_s, sigma_s = schedule.coefficients(boundary, sample)
    alpha_u, sigma_u = schedule.coefficients(time, sample)
    a = alpha_u / alpha_s
    # Valid u>s is enforced by phase_ranges and the sample-time sampler.
    b = (sigma_u.square() - a.square() * sigma_s.square()).clamp_min(1e-12).sqrt()
    return a, b, sigma_u


def renoise(schedule, sample, boundary, time, noise):
    a, b, _ = coefficients(schedule, boundary, time, sample)
    return a * sample + b * noise


def fake_loss(schedule, sample, boundary, time, noise, fake_epsilon, max_weight=10.):
    """Conditional score target epsilon*=sigma_u/b*noise; sample is detached.

    Move b out of the target and cap inverse-b^2 weights as in ambient training.
    Time-only weighting changes time emphasis, not the conditional optimum.
    At s=-1 and max_weight=inf this is exactly ordinary epsilon regression.
    """
    _, b, sigma_u = coefficients(schedule, boundary, time, sample.detach())
    target = (sigma_u * noise).detach()
    return ((b * fake_epsilon - target).square() * b.square().reciprocal().clamp_max(max_weight)).mean()


def dm_surrogate(schedule, sample, boundary, time, noised, fake_epsilon, teacher_epsilon):
    """Posterior-normalized reverse-KL direction, stopped before the surrogate.

    Raw dKL/dy = a*(score_F-score_T). Weight b^2/a^2/normalizer gives
    (posterior_mean_F-posterior_mean_T)/normalizer. At clean s=-1 it reduces
    to DMD2's clean-prediction direction, with its 0.5 MSE convention.
    """
    with torch.no_grad():
        a, b, sigma_u = coefficients(schedule, boundary, time, sample)
        score_f, score_t = -fake_epsilon / sigma_u, -teacher_epsilon / sigma_u
        posterior_t = (noised + b.square() * score_t) / a
        scale = (sample - posterior_t).abs().flatten(1).mean(1).clamp_min(1e-5)
        scale = scale.reshape(-1, *([1] * (sample.ndim - 1)))
        raw_direction = a * (score_f - score_t)
        weight = b.square() / a.square() / scale
        direction = weight * raw_direction
        target = (sample - direction).detach()
    return 0.5 * (sample - target).square().mean(), direction


@torch.no_grad()
def rollout_boundary_and_final(predict, schedule, condition, noise, knots, stage, both_middle=False):
    x = noise
    for k in range(1, 5):
        t = torch.full((len(x),), knots[k-1], device=x.device, dtype=torch.long)
        x = schedule.jump(x, predict(x, t, condition), t, knots[k])
        if k == stage:
            selected = x
        if both_middle and k == 5-stage:
            other = x
    return (selected, x, other) if both_middle else (selected, x)


def gan_refinement(predict, schedule, terminal, condition, timestep, noise):
    """Flash-DMD eqs. 8–9: only the extra low-noise student call receives GAN gradients."""
    time = torch.full((len(terminal),), timestep, device=terminal.device, dtype=torch.long)
    x = schedule.noise(terminal.detach(), noise.detach(), time)
    return schedule.clean(x, predict(x, time, condition), time)


def local_rollout(predict, schedule, condition, noise, knots, middle):
    """Yield live local outputs; consumer backwards BEFORE requesting next step.

    One four-call trajectory. No optimizer step until all local/PCM backwards.
    This intentionally does not implement two-step terminal unrolling.
    """
    selected = (middle,) if isinstance(middle, int) else tuple(middle)
    if selected not in ((2,), (3,), (2,3)):
        raise ValueError('Middle phase must be 2 or 3')
    x = noise.detach()
    for k in range(1, 5):
        t = torch.full((len(x),), knots[k-1], device=x.device, dtype=torch.long)
        with torch.set_grad_enabled(k in (*selected, 4)):
            x = schedule.jump(x, predict(x, t, condition), t, knots[k])
        if k in (*selected, 4):
            yield k, x
        x = x.detach()


def add_fake_stage_condition(model):
    """Standard UNet class conditioning, zero initialized to preserve base scores."""
    width = model.time_embedding.linear_2.out_features
    model.class_embedding = torch.nn.Embedding(3, width).to(
        device=next(model.parameters()).device, dtype=next(model.parameters()).dtype)
    torch.nn.init.zeros_(model.class_embedding.weight)
    model.register_to_config(num_class_embeds=3, class_embed_type=None,
        _use_default_values=[k for k in model.config.get("_use_default_values", [])
                             if k not in ("num_class_embeds", "class_embed_type")])


def stage_labels(stage, sample):
    if stage not in (2, 3, 4):
        raise ValueError('Overlapping fake scores require boundary stage 2, 3 or 4')
    return torch.full((len(sample),), stage-2, device=sample.device, dtype=torch.long)
