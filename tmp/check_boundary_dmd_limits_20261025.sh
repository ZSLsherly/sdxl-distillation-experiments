set -eu
source /data5/shenglin.zeng/env.sh
python - <<'PY'
import sys,json,os
from pathlib import Path
import torch
print('runtime',sys.executable,torch.__version__)
cg=next(line.split(':',2)[2].strip() for line in Path('/proc/self/cgroup').read_text().splitlines() if line.split(':')[1]=='memory')
p=Path('/sys/fs/cgroup/memory')/cg.lstrip('/')
while p!=Path('/sys/fs/cgroup'):
 info={}
 for f in ['memory.limit_in_bytes','memory.usage_in_bytes','memory.failcnt']:
  if (p/f).exists():info[f]=(p/f).read_text().strip()
 print(str(p),info)
 if p==Path('/sys/fs/cgroup/memory'):break
 p=p.parent
root=Path('/data5/shenglin.zeng/sdxl_crossdistill')
d=torch.load(root/'datasets/parallel_10k_seed20261018/prepared.pt',weights_only=True,mmap=True,map_location='cpu')
print('data',d['train_count'],list(d['latents'].shape))
print('base_present',(root/'models/sdxl-base-1.0/unet/config.json').exists())
PY
nvidia-smi --query-gpu=index,uuid,memory.free,utilization.gpu --format=csv,noheader,nounits
