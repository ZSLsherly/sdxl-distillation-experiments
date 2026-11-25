set -e
source /data5/shenglin.zeng/env.sh
df -h /data5
du -x -h --max-depth=1 /data5/shenglin.zeng 2>/dev/null
du -x -h --max-depth=2 /data5/shenglin.zeng/sdxl_crossdistill/outputs 2>/dev/null | sort -h
python - <<'PY'
import json
from pathlib import Path
r=Path('/data5/shenglin.zeng/sdxl_crossdistill')
print('ACTIVE',(r/'outputs/active_experiment.json').read_text())
for parent in ['gan_stage_parallel','gan_stage_joint34','dmd_control','gan_stage_v4','gan_stage_v3']:
 for p in (r/'outputs'/parent).glob('*'):
  if not p.is_dir():continue
  weights=[]
  for q in (p/'weights').glob('*'):
   if not q.is_dir():continue
   marker=q/'complete.json'
   meta=json.loads(marker.read_text()) if marker.exists() else {}
   total=sum(f.stat().st_size for f in q.rglob('*') if f.is_file())
   weights.append({'directory':str(q),'g':meta.get('student_updates'),'complete':marker.exists(),'bytes':total})
  if weights:print('WEIGHTS',json.dumps({'run':str(p),'count':len(weights),'bytes':sum(x['bytes'] for x in weights),'items':weights}))
PY
