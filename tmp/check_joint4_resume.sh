set -e
source /data5/shenglin.zeng/env.sh
cd /data5/shenglin.zeng/sdxl_crossdistill
python - <<'PY'
import json
from pathlib import Path
root=Path('outputs/gan_stage_parallel/joint4_20261017T232317_mb16')
for name in ['status.json','config.json']:
 print(name,(root/name).read_text())
for p in sorted(root.rglob('*')):
 if p.is_file() and (p.suffix in ('.pt','.pth','.safetensors') or p.name=='complete.json'):
  print(str(p),p.stat().st_size)
  if p.name=='complete.json':print(p.read_text())
print('ACTIVE',Path('outputs/active_experiment.json').read_text())
PY
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
