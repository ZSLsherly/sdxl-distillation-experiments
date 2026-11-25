source /data5/shenglin.zeng/env.sh
python - <<'PY'
import json,statistics
from pathlib import Path
root=Path('/data5/shenglin.zeng/sdxl_crossdistill/outputs/hparam_pilot')
for folder in sorted(root.glob('lr*')):
 file=folder/'rank0.jsonl'
 if not file.exists():continue
 rows=[json.loads(x) for x in file.read_text().splitlines()]
 student=[r for r in rows if r['phase']=='student'];fake=[r for r in rows if r['phase']=='fake_score']
 print(folder.name,'updates',len(rows),'student',len(student),'fake',len(fake),'student_median',round(statistics.median(r['seconds'] for r in student),3) if student else None)
 print('gradient_range', min((r['grad_norm'] for r in rows),default=0),max((r['grad_norm'] for r in rows),default=0),'clipped',sum(r['clip_scale']<1 for r in rows))
print('GPU')
PY
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
