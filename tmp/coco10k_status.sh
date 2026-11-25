set -eu
source /data5/shenglin.zeng/env.sh
cd /data5/shenglin.zeng/sdxl_crossdistill
python - <<'PY'
from pathlib import Path
import json
p=Path('outputs/coco10k_g160_20261020')
print('REAL_COUNT',len(list((p/'real').glob('*.jpg'))),'MANIFEST',(p/'manifest.json').exists())
for name in ['prepare.exit','prepare.log','inception_download.log','verify.exit','verify.log','run.exit','metrics.log','scores.json']:
    f=p/name
    if f.exists():
        text=f.read_text()
        print(name,text[-2200:])
for f in sorted(p.glob('status_*.json')):
    print(f.name,f.read_text())
PY
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
