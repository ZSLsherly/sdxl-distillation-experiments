set -euo pipefail
source /data5/shenglin.zeng/env.sh
cd /data5/shenglin.zeng/sdxl_crossdistill
python - <<'PY'
from pathlib import Path
import os,json,time,signal
out=Path('outputs/joint34_uniform_10k_2gpu_20261025')
(out/'USER_STOP').write_text('User requested stop training, queue, and automatic evaluation. Do not resume without a new request.\n')
pid=int((out/'workflow.pid').read_text())
proc=Path('/proc')/str(pid)
owned={}
if proc.exists():
 cmd=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
 assert 'joint34_a100_workflow_20261025.py' in cmd,(pid,'PID no longer belongs to this workflow')
 todo=[pid]
 while todo:
  p=todo.pop();base=Path('/proc')/str(p)
  try:
   stat=(base/'stat').read_text().rsplit(')',1)[1].split();owned[p]=stat[19]
   todo.extend(int(x) for x in (base/'task'/str(p)/'children').read_text().split())
  except (OSError,ProcessLookupError):pass
 for p in sorted(owned,reverse=True):
  try:os.kill(p,signal.SIGTERM)
  except ProcessLookupError:pass
 time.sleep(5)
 for p,start in owned.items():
  try:
   stat=(Path('/proc')/str(p)/'stat').read_text().rsplit(')',1)[1].split()
   if stat[19]==start and stat[0]!='Z':os.kill(p,signal.SIGKILL)
  except (OSError,ProcessLookupError):pass
record={'time':time.time(),'reason':'User requested stop. Training, queue and automatic evaluation cancelled. Do not resume automatically.','own_pids_signalled':sorted(owned),'user_requested':True}
(out/'workflow_stopped.json').write_text(json.dumps(record,indent=2))
(out/'queue_status.json').write_text(json.dumps({'state':'cancelled','time':time.time(),'reason':'user requested stop'},indent=2))
with (out/'workflow.jsonl').open('a') as f:f.write(json.dumps({'event':'stopped',**record})+'\n')
print(json.dumps(record))
PY
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader
