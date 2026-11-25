set -e
source /data5/shenglin.zeng/env.sh
cd /data5/shenglin.zeng/sdxl_crossdistill
python - <<'PY'
import json,os,signal,time
from pathlib import Path
from datetime import datetime
root=Path.cwd()
active=root/'outputs/active_experiment.json'
m=json.loads((active if active.exists() else root/'outputs/gan_stage_parallel/launch.json').read_text())
runs={r['run']:r for r in m['runs']}
def matching():
 found={}
 for p in Path('/proc').iterdir():
  if not p.name.isdigit():continue
  try:
   if p.stat().st_uid!=os.getuid():continue
   args=(p/'cmdline').read_bytes().decode().split('\0')
   if 'sdxl_parallel/train.py' not in args or '--run' not in args:continue
   run=args[args.index('--run')+1]
   if run in runs and (p/'cwd').resolve()==root:found[int(p.name)]=run
  except (FileNotFoundError,PermissionError,ProcessLookupError,ValueError):pass
 return found
before=matching()
print('STOP_TARGETS',before,flush=True)
for pid in before:
 try:os.kill(pid,signal.SIGTERM)
 except ProcessLookupError:pass
for _ in range(15):
 if not matching():break
 time.sleep(1)
remaining=matching()
for pid in remaining:
 try:os.kill(pid,signal.SIGKILL)
 except ProcessLookupError:pass
if remaining:time.sleep(2)
after=matching()
print('REMAINING',after,flush=True)
assert not after,after
for name,r in runs.items():
 p=root/'outputs'/name
 status=json.loads((p/'status.json').read_text())
 if name in before.values():
  record={'reason':'user_requested_stop','timestamp':datetime.now().astimezone().isoformat(),'terminated_pids':[pid for pid,run in before.items() if run==name],'last_training_status':status,'remaining_processes':[]}
  (p/'user_stop.json').write_text(json.dumps(record,indent=2))
  status.update(status='stopped_by_user',stopped_at=record['timestamp'])
  (p/'status.json').write_text(json.dumps(status,indent=2))
 print('FINAL',r['variant'],json.dumps(status),flush=True)
 for sub in ['eval','checkpoints','snapshots']:
  q=p/sub
  if q.exists():print('LAST_SAVED',r['variant'],sub,[x.name for x in sorted(q.iterdir())[-3:]],flush=True)
PY
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader
