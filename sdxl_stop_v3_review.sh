set -e
source /data5/shenglin.zeng/env.sh
cd /data5/shenglin.zeng/sdxl_crossdistill
python - <<'PY'
import os,signal,json,time
from pathlib import Path
root=Path('outputs/gan_stage_v3');launch=json.loads((root/'launch.json').read_text());runs={launch[v]['run'] for v in ['joint4','gan4']}
workers=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  args=(p/'cmdline').read_bytes().split(b'\0');decoded=[x.decode(errors='replace') for x in args]
  if '--run' in decoded and decoded[decoded.index('--run')+1] in runs and any(x.endswith('sdxl_gan_train_v3.py') for x in decoded):workers.append(int(p.name))
 except (FileNotFoundError,PermissionError,ValueError,IndexError):pass
for pid in workers:
 try:os.kill(pid,signal.SIGTERM)
 except ProcessLookupError:pass
launchers=[]
for v in ['joint4','gan4']:
 pid=launch[v]['launcher_pid']
 try:
  cmd=Path(f'/proc/{pid}/cmdline').read_text().replace('\0',' ')
  assert 'sdxl_gan_train_v3.py' in cmd and os.getpgid(pid)==pid,(pid,cmd)
  os.killpg(pid,signal.SIGTERM);launchers.append(pid)
 except ProcessLookupError:pass
 except FileNotFoundError:pass
time.sleep(3)
remaining=[]
for pid in workers:
 try:
  cmd=Path(f'/proc/{pid}/cmdline').read_bytes()
  if b'sdxl_gan_train_v3.py' in cmd:os.kill(pid,signal.SIGKILL);remaining.append(pid)
 except FileNotFoundError:pass
record={'reason':'Stopped for v3 code review fixes; no completed optimizer updates yet; existing files retained','workers_terminated':workers,'launcher_groups_terminated':launchers,'force_killed_if_still_exiting':remaining,'old_launch':launch}
(root/'review_stop_20261017.json').write_text(json.dumps(record,indent=2))
for v in ['joint4','gan4']:
 p=Path('outputs')/launch[v]['run'];s=json.loads((p/'status.json').read_text()) if (p/'status.json').exists() else {};s['status']='stopped_for_code_review';s['reason']=record['reason'];(p/'status.json').write_text(json.dumps(s,indent=2))
(root/'launch_before_code_review.json').write_text(json.dumps(launch,indent=2));(root/'launch.json').unlink()
print(json.dumps({'stopped_workers':workers,'stopped_launchers':launchers}))
PY
