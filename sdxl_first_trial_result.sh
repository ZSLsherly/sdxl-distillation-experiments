source /data5/shenglin.zeng/env.sh
python - <<'PY'
import json,statistics
from pathlib import Path
r=Path('/data5/shenglin.zeng/sdxl_crossdistill/outputs/hparam_pilot/lr5e7_pcm10')
s=json.loads((r/'summary.json').read_text());print(json.dumps(s,indent=2))
for phase in ['before','after']:
 rows=[]
 for rank in range(4):rows+=json.loads((r/f'validation_{phase}_rank{rank}.json').read_text())
 print(phase, {k:statistics.mean(x[k] for x in rows) for k in ['boundary_relative_mse','final_teacher_pair_rmse','latent_mean','latent_std']})
PY
