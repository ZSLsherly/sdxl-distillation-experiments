set -eu
source /data5/shenglin.zeng/env.sh
cd /data5/shenglin.zeng/sdxl_crossdistill
python - <<'PY'
from pathlib import Path
import json
p=Path('outputs/coco10k_g160_20261020')
for name in ['verify.exit','run.exit','metrics.log','scores.json','scores_ready.json']:
    f=p/name
    if f.exists():print(name,f.read_text()[-1800:])
for f in sorted(p.glob('status_*.json')):
    print(f.name,f.read_text())
for f in sorted(p.glob('worker_?.log')):
    print(f.name,f.read_text()[-700:])
for d in sorted((p/'features').glob('*')):
    total=0
    for f in d.glob('*.npz'):
        a,b=map(int,f.stem.split('_'));total+=b-a
    print('COMMITTED',d.name,total)
PY
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
