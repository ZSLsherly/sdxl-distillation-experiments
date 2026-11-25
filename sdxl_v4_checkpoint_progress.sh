set -e
source /data5/shenglin.zeng/env.sh
cd /data5/shenglin.zeng/sdxl_crossdistill
python - <<'PY'
from pathlib import Path
import json,statistics
p=Path('outputs/v4_validation/latest_runs.txt');run=Path('outputs')/p.read_text().splitlines()[0]
for d in (run/'checkpoints').glob('*'):
 print('CHECKPOINT',d.name,[(f.name,round(f.stat().st_size/1024**3,2)) for f in d.glob('*.pt')])
paths=sorted((run/'eval').glob('*_g000001_*'))
if paths:
 rows=[]
 for f in paths[-1].glob('rank*.json'):rows.extend(json.loads(f.read_text())['samples'])
 print('DIAGNOSTIC_IMAGES',len(rows))
 for kind in ['real','fake']:
  v=[x['decode_comparison'][kind]['rgb01_mae'] for x in rows]
  if v:print('TILE_FULL_MAE',kind,statistics.mean(v))
PY
free -h
df -h /data5
