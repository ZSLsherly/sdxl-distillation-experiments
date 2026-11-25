"""Student boundary-DM weights and a fixed-batch, pre-clip gradient diagnostic."""
import torch


def student_dm_term(dm, stage, both_middle, terminal_weight):
    if stage == 4:
        return dm if terminal_weight == 1 else terminal_weight * dm
    return 0.5 * dm if both_middle else dm


def gradient_probe(dm_loss, pcm_loss, clear_grad, grad_norm, weight, pcm_weight, device, seed):
    """Measure full shared-parameter gradients without a second UNet-sized buffer."""
    devices = [device] if torch.device(device).type == 'cuda' else []

    def backward_once(loss_fn, coeff):
        # Replay identical dropout and preserve the training RNG state.
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            (coeff * loss_fn()).backward()

    clear_grad()
    try:
        backward_once(dm_loss, weight)
        dm = grad_norm()
        clear_grad()
        backward_once(pcm_loss, pcm_weight)
        pcm = grad_norm()
        backward_once(dm_loss, weight)
        total = grad_norm()
    finally:
        clear_grad()

    # 2 <p,d> = ||p+d||^2 - ||p||^2 - ||d||^2.
    cosine = None if pcm * dm == 0 else (total**2 - pcm**2 - dm**2) / (2 * pcm * dm)
    return dict(pcm_grad_norm=pcm, dm4_weighted_grad_norm=dm,
                dm4_raw_grad_norm=dm / weight, joint_grad_norm=total,
                weighted_dm_over_pcm=dm / pcm if pcm else None,
                cosine=None if cosine is None else max(-1., min(1., cosine)))
