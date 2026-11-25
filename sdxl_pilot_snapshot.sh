source /data5/shenglin.zeng/env.sh
root=/data5/shenglin.zeng/sdxl_crossdistill
tail -n 6 "$root/logs/stage_metrics.log"
tail -n 2 "$root/logs/hparam_pipeline.log"
python - <<'PY'
import json,statistics
from pathlib import Path
root=Path('/data5/shenglin.zeng/sdxl_crossdistill/outputs/hparam_pilot')
for p in sorted(root.glob('lr*')):
 q=p/'rank0.jsonl'
 if not q.exists():continue
 rows=[json.loads(s) for s in q.read_text().splitlines()];students=[r for r in rows if r['phase']=='student']
 print(p.name,'steps',len(rows),'student',len(students),'peak_GiB',max((r['peak_allocated_GiB'] for r in rows),default=0))
 if students: print('last_student',students[-1])
PY
