source /data5/shenglin.zeng/env.sh
cd /data5/shenglin.zeng/sdxl_crossdistill
python - <<'PY'
from pathlib import Path
import json,datetime
run=Path('outputs')/Path('outputs/v4_validation/latest_runs.txt').read_text().splitlines()[0]
print(datetime.datetime.now().isoformat())
for f in run.glob('pid_rank*'):
 pid=f.read_text().strip();p=Path('/proc')/pid
 if p.exists():
  print(pid,'wchan', (p/'wchan').read_text().strip())
  print((p/'io').read_text())
print('manifests',list((run/'checkpoints').glob('*/complete.json')))
PY
tail -n 6 logs/v4_minimal_supervisor.log
