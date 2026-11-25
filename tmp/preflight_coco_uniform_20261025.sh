set -euo pipefail
source /data5/shenglin.zeng/env.sh
cd /data5/shenglin.zeng/sdxl_crossdistill
date -Is
nvidia-smi --query-gpu=index,uuid,name,memory.free,utilization.gpu --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv
python - <<'PY'
from pathlib import Path
import json
for p in Path('outputs').iterdir():
 if p.is_dir() and any(w in p.name for w in ['coco','eval']):
  print('EVAL_DIR',p)
  for name in ('manifest.json','scores.json','complete.json'):
   f=p/name
   if f.exists():
    x=json.loads(f.read_text())
    if name=='manifest.json':print(name,{k:x.get(k) for k in ['n','source_zip_sha256','csv_sha256','protocol']},'models',list(x.get('models',{})))
    else:print(name,x)
base=Path('outputs/joint4_uniform_10k_2gpu_mb16_20261023/train')
for p in (base/'weights').iterdir():
 f=p/'complete.json'
 if f.exists() and json.loads(f.read_text())['student_updates']==160:print('G160_CHECKPOINT',p)
PY
