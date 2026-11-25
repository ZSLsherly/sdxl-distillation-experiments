source /data5/shenglin.zeng/env.sh
cd /data5/shenglin.zeng/sdxl_crossdistill
python - <<'PY'
import json
from pathlib import Path
active=Path('outputs/active_experiment.json')
m=json.loads((active if active.exists() else Path('outputs/gan_stage_parallel/launch.json')).read_text())
print(json.dumps(m,indent=2))
for run in m['runs']:
 p=Path('outputs')/run['run']
 print('RUN',run['variant'])
 if (p/'status.json').exists(): print((p/'status.json').read_text())
 f=Path(run['log']) if 'log' in run else Path('logs')/f"parallel_{run['variant']}_{m['timestamp']}.log"
 if f.exists():
  for line in f.read_text(errors='replace').splitlines()[-8:]: print(line[:500])
 if (p/'rank0.jsonl').exists():
  for line in (p/'rank0.jsonl').read_text().splitlines()[-2:]:
   r=json.loads(line)
   print({k:v for k,v in r.items() if k in ('phase','g','fake_update','seconds','cycle_seconds','fake_loss','D_hinge','grad_norm','peak_allocated_GiB')})
PY
nvidia-smi --query-gpu=index,memory.used,utilization.gpu,power.draw --format=csv,noheader
