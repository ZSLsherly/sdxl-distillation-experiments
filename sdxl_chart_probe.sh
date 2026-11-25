set -e
source /data5/shenglin.zeng/env.sh
cd /data5/shenglin.zeng/sdxl_crossdistill
python - <<'PY'
import json,importlib.util
from pathlib import Path
for m in ['pyiqa','piq','scipy','cv2','torchmetrics','timm']:print('MODULE',m,bool(importlib.util.find_spec(m)))
r=json.loads(Path('outputs/gan_stage_parallel/launch.json').read_text())
for t in r['runs']:
 p=Path('outputs')/t['run'];e=sorted((p/'eval').glob('*'));complete=[]
 for d in e:
  fs=list(d.glob('rank*.json'))
  if len(fs)==2:
   rows=[x for f in fs for x in json.loads(f.read_text())['samples']]
   if len(rows)==8 and all((d/x['image']).exists() for x in rows):complete.append((d.name,rows))
 print('EVAL',t['variant'],len(complete),'LATEST',complete[-1][0])
 if complete:
  for row in complete[0][1]:print('CASE',t['variant'],row['split'],row['original_index'],row['prompt'])
for p in Path('models').glob('**/*'):
 if p.is_file() and any(x in str(p).lower() for x in ['metric','niqe','musiq','reward','hps']):print('MODEL',p)
PY
free -h
