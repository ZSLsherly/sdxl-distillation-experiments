set -eu
R=/data1/shenglin.zeng/dmd2_official_recipe_4gpu_20261123
printf '%s\n' '2026-11-23 explicit user instruction: stop DMD2 and downstream evaluation.' > "$R/STOP_REQUEST"
sleep 8
"$R/env/bin/python" - <<'PY'
from pathlib import Path
import json,os,signal,time
r=Path('/data1/shenglin.zeng/dmd2_official_recipe_4gpu_20261123')
targets={str(r/'code'/n) for n in ['run_guard.py','train_matched.py']}
targets.update(str(r/'coco10k/code'/n) for n in ['run_queue.py','evaluate_dmd2.py'])
def own():
 found=[]
 for p in Path('/proc').iterdir():
  if not p.name.isdigit():continue
  try:argv=(p/'cmdline').read_bytes().decode(errors='replace').split('\0')
  except OSError:continue
  if targets.intersection(argv):found.append(int(p.name))
 return found
remaining=own()
for pid in remaining:
 try:os.kill(pid,signal.SIGTERM)
 except ProcessLookupError:pass
if remaining:time.sleep(5)
for pid in own():
 try:os.kill(pid,signal.SIGKILL)
 except ProcessLookupError:pass
time.sleep(1)
assert not own(),'Own DMD2 processes still present'
p=r/'runs/formal_g160/status.json'
s=json.loads(p.read_text()) if p.exists() else {}
s.update(status='stopped_by_user',stop_reason='Explicit user request 2026-11-23; downstream evaluation cancelled')
p.write_text(json.dumps(s,indent=2))
(r/'用户停止记录_20261123.txt').write_text('用户明确要求停止DMD2。仅终止该独立目录的训练和评测进程，STOP_REQUEST保留阻止后续队列。\n'+json.dumps(s,ensure_ascii=False,indent=2))
print('DMD2_STOPPED',json.dumps(s))
PY
nvidia-smi -i 1,2,3,5 --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
