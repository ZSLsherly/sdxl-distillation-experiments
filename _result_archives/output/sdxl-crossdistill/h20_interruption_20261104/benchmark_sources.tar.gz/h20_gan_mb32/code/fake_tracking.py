"""Optional single-GPU pilot: move shared fake parameters towards the updated student."""
import math
import torch
from torch.distributed._composable.fsdp import FSDPModule
import boundary


def verify_tracking_probes(rows, ids, updates, start, rate):
    expected = [g for g in range(1, updates+1) if g % 8 == 0 or g == start or (rate and g == 121)]
    probes = [r for r in rows if r['phase'] == 'heldout_fake_tracking']
    if len(set(ids)) != 8 or len(ids) != 8 or [r['g'] for r in probes] != expected:
        raise ValueError("Incomplete held-out tracking cases or updates")
    for r in probes:
        if r['original_ids'] != ids or r['student_mix'] != (rate if r['g'] >= start else 0):
            raise ValueError("Tracking sample identity or mixing fraction changed")
        values = r['results']
        if [v['time'] for v in values] != [60, 499, 940] or not all(
            math.isfinite(v[k]) for v in values for k in ('before','after','direction_cosine','relative_direction_change')
        ):
            raise ValueError("Missing or nonfinite held-out tracking measurements")


@torch.no_grad()
def mix_shared_parameters(fake, student, rate):
    if not 0 < rate <= 1:
        raise ValueError("Expected a student mixing fraction in (0,1]")
    for model in (fake, student):
        for module in model.modules():
            if isinstance(module, FSDPModule):
                module.reshard()
    src, dst = dict(student.named_parameters()), dict(fake.named_parameters())
    if dst.keys() - src.keys() != {"class_embedding.weight"} or src.keys() - dst.keys():
        raise ValueError("Expected matching UNets and one fake-only stage embedding")
    for name, param in src.items():
        other = dst[name]
        left = other.to_local() if hasattr(other, "to_local") else other
        right = param.to_local() if hasattr(param, "to_local") else param
        if left.shape != right.shape or left.dtype != right.dtype or left.device != right.device:
            raise ValueError(f"Incompatible model parameters: {name}")
        left.lerp_(right, rate)


@torch.no_grad()
def terminal_probe(schedule, clean, score_fake, score_teacher, mix, seed):
    """Hold the terminal latent, t, eps and teacher fixed across one parameter blend."""
    gen = torch.Generator(device=clean.device).manual_seed(seed)
    eps = torch.randn(clean.shape, device=clean.device, dtype=clean.dtype, generator=gen)
    def measure(t, xt, teacher):
        pred = score_fake(xt, t)
        loss = boundary.fake_loss(schedule, clean, -1, t, eps, pred, 10.)
        direction = boundary.dm_surrogate(schedule, clean, -1, t, xt, pred, teacher)[1]
        return loss.item(), direction.flatten(1)
    cases = []
    for time in (60, 499, 940):
        t = torch.full((len(clean),), time, device=clean.device, dtype=torch.long)
        xt = boundary.renoise(schedule, clean, -1, t, eps)
        teacher = score_teacher(xt, t)
        loss, direction = measure(t, xt, teacher)
        cases.append((time, t, xt, teacher, loss, direction))
    mix()
    results = []
    for time,t,xt,teacher,before,old in cases:
        after,new = measure(t, xt, teacher)
        cosine = torch.nn.functional.cosine_similarity(old,new,dim=1).mean().item()
        change = ((new-old).norm(dim=1)/old.norm(dim=1).clamp_min(1e-8)).mean().item()
        if not all(math.isfinite(v) for v in (before,after,cosine,change)):
            raise FloatingPointError("Non-finite held-out fake diagnostic")
        results.append(dict(time=time, before=before, after=after, direction_cosine=cosine,
                            relative_direction_change=change))
    return results
