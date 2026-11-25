source /data5/shenglin.zeng/env.sh
cd /data5/shenglin.zeng/sdxl_crossdistill
python - <<'PY'
import json
from pathlib import Path
p=Path('outputs/gan_stage_pilot/preflight')
s=json.loads((p/'summary.json').read_text())
rows=[json.loads(l) for l in (p/'rank0.jsonl').read_text().splitlines()]
print(json.dumps({'status':s['status'],'student_updates':s['student_updates'],'fake_updates':s['fake_updates'],'train_seconds':s['train_seconds'],'peak_allocated_GiB':s['peak_allocated_GiB'],'peak_reserved_GiB':s['peak_reserved_GiB'],'own_process_peak_MiB':s.get('nvidia_smi_peak_own_process_MiB_per_gpu'),'student_rows':[r for r in rows if r['phase']=='student'],'R1_exercised':any(r.get('r1',0)>0 for r in rows)},indent=2))
PY
tail -n 8 logs/gan_stage_pilot.log
if test -f logs/gan_assessment.log; then tail -n 3 logs/gan_assessment.log; fi
if test -f logs/gan_dmd.log; then tail -n 4 logs/gan_dmd.log; fi
