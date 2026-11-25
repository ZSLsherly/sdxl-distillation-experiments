source /data5/shenglin.zeng/env.sh
cd /data5/shenglin.zeng/sdxl_crossdistill
python - <<'PY'
import json,statistics,math
from pathlib import Path
from datetime import datetime
m=json.loads(Path('outputs/active_experiment.json').read_text());p=Path('outputs')/m['runs'][0]['run']
rows=[json.loads(s) for s in (p/'rank0.jsonl').read_text().splitlines()]
gs=[r for r in rows if r['phase']=='student_done'];last=gs[-12:]
wall=(datetime.strptime(last[-1]['timestamp'],'%Y%m%dT%H%M%S%z')-datetime.strptime(last[0]['timestamp'],'%Y%m%dT%H%M%S%z')).total_seconds()/(last[-1]['g']-last[0]['g'])
ev=[]
for d in sorted((p/'eval').iterdir()):
 fs=[d/f'rank{i}.json' for i in range(4)]
 if all(f.exists() for f in fs):
  ds=[json.loads(f.read_text()) for f in fs];rs=[r for x in ds for r in x['samples']]
  if len(rs)==8 and all((d/r['image']).exists() for r in rs):ev.append(ds[0]['student_updates'])
weights=[json.loads(f.read_text()) for f in sorted((p/'weights').glob('*/complete.json'))]
print(json.dumps({'checked_at':datetime.now().astimezone().isoformat(),'completed_g':gs[-1]['g'],'recent_mean_training_seconds':statistics.mean(r['cycle_seconds'] for r in last),'recent_wall_seconds_per_g':wall,'remaining_minutes_estimate':(160-gs[-1]['g'])*wall/60,'completed_evals':len(ev),'last_eval_g':max(ev),'saved_weights_g':[r['student_updates'] for r in weights],'recent_losses_finite':all(math.isfinite(r[k]) for r in last for k in ['dmd_loss','pcm_loss','grad_norm']),'last_student':{k:gs[-1][k] for k in ['g','stage','dmd_loss','pcm_loss','grad_norm','gan','timestamp']}},indent=2))
PY
