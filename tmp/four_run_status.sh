source /data5/shenglin.zeng/env.sh
cd /data5/shenglin.zeng/sdxl_crossdistill
python - <<'PY'
import json
from pathlib import Path
runs=['dmd_control/dmd_only_20261018T141942_4gpu_mb16','gan_stage_parallel/gan4_20261017T232317_mb16','gan_stage_parallel/joint4_20261017T232317_mb16','gan_stage_joint34/joint34_20261018T091442_4gpu_mb16','gan_stage_joint4_4gpu/joint4_20261019T122237_4gpu_mb16']
for run in runs:
 p=Path('outputs')/run
 print('RUN',run)
 print('STATUS',(p/'status.json').read_text() if (p/'status.json').exists() else 'missing')
 f=p/'rank0.jsonl'
 if f.exists():
  rows=[json.loads(s) for s in f.read_text().splitlines()]
  gs=[r for r in rows if r['phase']=='student_done']
  if gs: print('LAST_G',json.dumps({k:gs[-1].get(k) for k in ['g','timestamp','cycle_seconds','dmd_loss','gan_loss','grad_norm']}))
 states=list((p/'training_state').glob('*/complete.json'))
 for q in states:print('FULL_STATE',str(q))
print('ACTIVE',Path('outputs/active_experiment.json').read_text())
PY
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
