set -e
source /data5/shenglin.zeng/env.sh
cd /data5/shenglin.zeng/sdxl_crossdistill
python - <<'PY'
import json
from pathlib import Path
r=json.loads(Path('outputs/gan_stage_v4/launch.json').read_text())
for v in ['joint4','gan4']:
 p=Path('outputs')/r[v]['run'];print('VARIANT',v)
 for f in sorted((p/'weights').glob('*/complete.json')):
  print('WEIGHT',f.parent.name,json.loads(f.read_text()))
 for d in sorted((p/'eval').glob('*')):
  rows=list(d.glob('rank*.json'));print('EVAL',d.name,'rank_jsons',len(rows),'images',len(list(d.glob('*.png'))))
 logs=p/'rank0.jsonl'
 if logs.exists():
  a=[json.loads(x) for x in logs.read_text().splitlines()]
  for x in [q for q in a if q['phase']=='student_done'][-5:]:print('STUDENT',{k:x[k] for k in ['timestamp','g','grad_norm','effective_batch']})
PY
