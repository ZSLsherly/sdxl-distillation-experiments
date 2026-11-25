source /data5/shenglin.zeng/env.sh
python - <<'PY'
import json,statistics
from pathlib import Path
root=Path('/data5/shenglin.zeng/sdxl_crossdistill/outputs/hparam_pilot')
for name in ['lr5e7_pcm10','lr25e8_pcm10','lr5e7_pcm1']:
 p=root/name;rows=[]
 for rank in range(4):rows+=json.loads((p/f'validation_after_rank{rank}.json').read_text())
 print(name,'boundary_relative_mse',statistics.mean(r['boundary_relative_mse'] for r in rows),'teacher_final_pair_rmse',statistics.mean(r['final_teacher_pair_rmse'] for r in rows))
PY
