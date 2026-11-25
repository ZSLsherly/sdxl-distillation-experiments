"""Replay fixed losses for pre-clip norms/cosines without storing gradient copies."""
import math
import random
import torch

PROBE_UPDATES = (1, 40, 80, 120, 160)


def gradient_diagnostics(losses, clear_grad, grad_norm, device, seed):
    # No optimizer call. Each replay restores RNG; caller supplies fixed inputs.
    devices = [torch.device(device).index or 0] if torch.device(device).type == 'cuda' else []
    py_state = random.getstate()

    def measure(names):
        clear_grad()
        for name in names:
            with torch.random.fork_rng(devices=devices):
                # Seed only the selected GPU; don't disturb other devices.
                torch.random.default_generator.manual_seed(seed)
                if devices:
                    with torch.cuda.device(devices[0]):
                        torch.cuda.manual_seed(seed)
                losses[name]().backward()
        value = float(grad_norm())
        if not math.isfinite(value):
            raise FloatingPointError('Non-finite diagnostic gradient')
        return value

    try:
        norms = {k: measure((k,)) for k in ('PCM', 'DM2', 'DM3', 'DM4')}
        cosines, combined = {}, {}
        for a,b in (('PCM','DM2'),('PCM','DM3'),('PCM','DM4'),('DM2','DM4'),('DM3','DM4')):
            ab=measure((a,b));combined[a+'+'+b]=ab
            denom=2*norms[a]*norms[b]
            cosine=(ab*ab-norms[a]**2-norms[b]**2)/denom if denom else None
            cosines[a+'_'+b]=None if cosine is None else max(-1.,min(1.,cosine))
        ratios={k:norms['PCM']/combined[k] if combined[k] else None for k in ('DM2+DM4','DM3+DM4')}
        ratios['DM4']=norms['PCM']/norms['DM4'] if norms['DM4'] else None
        return dict(grad_norms=norms, gradient_cosines=cosines, combined_grad_norms=combined,
                    pcm_over_dm_norm=ratios, gradients='pre-clip; fixed 8-sample diagnostic, not batch64')
    finally:
        clear_grad()
        random.setstate(py_state)
