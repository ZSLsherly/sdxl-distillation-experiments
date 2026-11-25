"""Query selected UUIDs directly; never enumerate the faulty GPU0."""
import os
from pathlib import Path
import pynvml as nv

nv.nvmlInit()
def status(uuid):
    h=nv.nvmlDeviceGetHandleByUUID(uuid)
    pids={p.pid for p in nv.nvmlDeviceGetComputeRunningProcesses(h)}
    # The verified gdm Xorg service uses 4.5 MiB on every card; it is not a job.
    for p in nv.nvmlDeviceGetGraphicsRunningProcesses(h):
        try:is_display=Path(f'/proc/{p.pid}/comm').read_text().strip()=='Xorg' and p.usedGpuMemory<64*2**20
        except OSError:is_display=False
        if not is_display:pids.add(p.pid)
    return pids,nv.nvmlDeviceGetMemoryInfo(h).free,nv.nvmlDeviceGetUtilizationRates(h).gpu
class ResourceConflict(RuntimeError):pass
def guard(uuid):
    if status(uuid)[0]-{os.getpid()}:raise ResourceConflict('Another process entered selected GPU')
