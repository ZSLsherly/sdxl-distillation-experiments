set -eu
. /root/sdxl_research/env.sh
free -h
for f in /sys/fs/cgroup/memory/memory.limit_in_bytes /sys/fs/cgroup/cpu/cpu.cfs_quota_us; do test ! -f "$f" || cat "$f"; done
python - <<'PY'
import torch
from pathlib import Path
print('TORCH',torch.__version__,'GPU',torch.cuda.get_device_name(0))
for p in ['models/sdxl-base-1.0/unet/config.json','datasets/parallel_10k_seed20261018/prepared.pt','code/sdxl_research/train.py']:
 q=Path('/root/sdxl_research')/p;print(p,q.exists(),q.stat().st_size)
PY
