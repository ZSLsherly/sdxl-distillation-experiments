set -e
source /data5/shenglin.zeng/env.sh
cd /data5/shenglin.zeng/sdxl_crossdistill
python - <<'PY'
from pathlib import Path
import os,signal,time,json
runs=set(Path('outputs/v4_validation/latest_runs.txt').read_text().splitlines())
# Stop only this validation supervisor, then its specifically named training runs.
for pid,needle in [(3142319,'sdxl_validation_minimal_supervisor.py'),(3140063,'sdxl_run_v4_validation.sh')]:
 try:
  if needle in Path(f'/proc/{pid}/cmdline').read_text():os.kill(pid,signal.SIGTERM)
 except (FileNotFoundError,ProcessLookupError):pass
pids=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  a=[x.decode(errors='replace') for x in (p/'cmdline').read_bytes().split(b'\0')]
  if '--run' in a and a[a.index('--run')+1] in runs and any(x.endswith('sdxl_gan_train_v4.py') for x in a):pids.append(int(p.name))
 except (FileNotFoundError,PermissionError,ValueError,IndexError):pass
for pid in pids:
 try:os.kill(pid,signal.SIGTERM)
 except ProcessLookupError:pass
time.sleep(2)
for pid in pids:
 try:
  if b'sdxl_gan_train_v4.py' in Path(f'/proc/{pid}/cmdline').read_bytes():os.kill(pid,signal.SIGKILL)
 except (FileNotFoundError,ProcessLookupError):pass
Path('outputs/v4_validation/stopped_for_simplification.json').write_text(json.dumps({'reason':'User requested lean research code; removed heavyweight full-state validation','stopped_pids':pids,'runs':list(runs)},indent=2))
print('VALIDATION_STOPPED',pids)
PY
