set -e
source /data5/shenglin.zeng/env.sh
cd /data5/shenglin.zeng/sdxl_crossdistill
python - <<'PY'
import json,statistics
from pathlib import Path
r=json.loads(Path('outputs/gan_stage_v4/launch.json').read_text())
for v in ['joint4','gan4']:
 p=Path('outputs')/r[v]['run'];dirs=sorted((p/'eval').glob('*'))
 for d in dirs:
  rows=[]
  for f in d.glob('rank*.json'):rows.extend(json.loads(f.read_text())['samples'])
  if len(rows)!=8:continue
  print(v,d.name)
  for split in ['train','holdout']:
   a=[x for x in rows if x['split']==split]
   print(split,json.dumps({'n':len(a),'laplacian_variance':statistics.mean(x['laplacian_variance_diagnostic_only'] for x in a),'D_real':statistics.mean(x['D']['real']['total'] for x in a),'D_fake':statistics.mean(x['D']['fake']['total'] for x in a)}))
PY
