set -eu
. /root/sdxl_research/env.sh
df -h "$SDXL_ROOT" "$SDXL_WORK"
free -h
for f in /sys/fs/cgroup/memory.max /sys/fs/cgroup/memory.current /sys/fs/cgroup/cpu.max; do if test -f "$f"; then cat "$f"; fi; done
find "$SDXL_ROOT/datasets" "$SDXL_WORK" -maxdepth 4 \( -name 'vae_latents_*.pt' -o -name 'prepared.pt' -o -name 'parallel_20k*' \) -printf '%p %s\n'
export CUDA_VISIBLE_DEVICES=''
python - <<'PY'
import os,torch
from pathlib import Path
o=Path(os.environ['SDXL_ROOT'])/'datasets/parallel_10k_seed20261018/prepared.pt'
v=torch.load(o,map_location='cpu',weights_only=True,mmap=True)
print('cache',[(k,(tuple(t.shape),str(t.dtype)) if torch.is_tensor(t) else (type(t).__name__,len(t) if isinstance(t,list) else t)) for k,t in v.items()])
print('cpu flags', 'amx_bf16' in Path('/proc/cpuinfo').read_text())
PY
