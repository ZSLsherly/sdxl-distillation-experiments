set -eu
root=/data5/shenglin.zeng/sdxl_crossdistill
date -Is
python3 - <<'PY'
from pathlib import Path
import json
root=Path('/data5/shenglin.zeng/sdxl_crossdistill')
for p in sorted((root/'outputs').rglob('config.json')):
 if 'code' in p.parts or 'models' in p.parts:continue
 try:d=json.loads(p.read_text())
 except Exception:continue
 if 'turning_sampling' in d or 'turning' in str(p).lower():
  print('RUN_CONFIG',str(p),json.dumps(d))
  for name in ['status.json','complete.json','workflow_error.json']:
   f=p.parent/name
   if f.exists():print(name,f.read_text())
  for name in ['metrics.jsonl','train.jsonl','training.jsonl','events.jsonl']:
   f=p.parent/name
   if f.exists():
    lines=f.read_text().splitlines();print('LOG',name,'lines',len(lines),'TAIL',lines[-2:])
for p in sorted((root/'outputs').glob('*turning*')):print('TURNING_DIR',p)
for p in sorted((root/'logs').glob('*turning*')):
 print('TURNING_LOG',p)
 print('\n'.join(p.read_text(errors='replace').splitlines()[-8:]))
PY
nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader,nounits
