"""Small execution helpers; no change to objectives or sample selection."""
from collections import defaultdict
from contextlib import nullcontext
import torch
import torch.distributed as dist


@torch.no_grad()
def average_gradients(parameters, bucket_mb=16, world_size=None, reduce=None):
    """Reduce same-dtype/device buckets; preserve per-parameter mean gradients.

    All ranks must use the same trainable parameter set, as in our replicated D.
    A nonpositive cap selects the original one-collective-per-parameter path.
    """
    world_size = dist.get_world_size() if world_size is None else world_size
    reduce = dist.all_reduce if reduce is None else reduce
    groups = defaultdict(list)
    for parameter in parameters:
        if parameter.grad is not None:
            groups[(parameter.grad.device, parameter.grad.dtype)].append(parameter.grad)
    calls = 0
    for grads in groups.values():
        if bucket_mb <= 0:
            for grad in grads:
                reduce(grad); grad.div_(world_size); calls += 1
            continue
        bucket, size = [], 0
        def flush():
            flat = torch.cat([g.reshape(-1) for g in bucket])
            reduce(flat); flat.div_(world_size)
            offset = 0
            for g in bucket:
                g.copy_(flat[offset:offset + g.numel()].view_as(g))
                offset += g.numel()
        for grad in grads:
            nbytes = grad.numel() * grad.element_size()
            if bucket and size + nbytes > bucket_mb * 2**20:
                flush(); calls += 1; bucket, size = [], 0
            bucket.append(grad); size += nbytes
        if bucket:
            flush(); calls += 1
    return calls


class TraceWindow:
    """Optional first-N-cycle PyTorch trace. Off means no profiler overhead."""
    def __init__(self, cycles, path, enabled):
        self.remaining = cycles if enabled else 0
        self.path = path
        self.profiler = None
    def start(self):
        if self.remaining:
            self.profiler = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                record_shapes=True, profile_memory=True, with_stack=False)
            self.profiler.start()
    def region(self, name):
        return torch.profiler.record_function(name) if self.profiler else nullcontext()
    def step(self):
        if self.profiler:
            self.profiler.step(); self.remaining -= 1
            if not self.remaining:
                self.close()
    def close(self):
        if self.profiler:
            self.profiler.stop()
            self.profiler.export_chrome_trace(str(self.path))
            self.profiler = None


def schedule_settings(schedule, dmd_t_max=None, pcm_boundary=None):
    if len(schedule) != 4 or schedule[0] != 999 or not all(a > b >= 0 for a,b in zip(schedule,schedule[1:])):
        raise ValueError('Expected four descending timesteps starting at 999')
    boundary = schedule[1] if pcm_boundary is None else pcm_boundary
    # DMD re-noising is independent of inference stage boundaries.
    upper = 919 if dmd_t_max is None else dmd_t_max
    if not 20 <= upper <= 999 or not 0 <= boundary <= 978:
        raise ValueError('Invalid DMD/PCM support')
    return list(schedule) + [-1], boundary, upper
