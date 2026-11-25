source /data5/shenglin.zeng/env.sh
python - <<'PY'
import os,subprocess,json
from pathlib import Path
print('NOW',subprocess.check_output(['date','--iso-8601=seconds'],text=True).strip())
print(subprocess.check_output(['ps','-p','3575376,3574011,3598411','-o','user:16,pid,lstart,etime,args'],text=True))
for pid in [3575376,3574011,3598411]:
 p=Path('/proc')/str(pid)
 print('PID',pid)
 try:
  cmd=(p/'cmdline').read_bytes().decode().split('\0')
  scripts=[s for s in cmd if s.endswith('.py')]
  try:cwd=(p/'cwd').resolve(strict=True);print('CWD',str(cwd))
  except OSError as e:cwd=None;print('CWD_ACCESS',type(e).__name__)
  for s in scripts:
   f=Path(s)
   if not f.is_absolute():
    if cwd is None:print('RELATIVE_SCRIPT',s,'cannot resolve cwd');continue
    f=cwd/f
   try:
    with f.open('rb') as h:h.read(1)
    print('SCRIPT_READABLE',str(f),'bytes',f.stat().st_size)
   except OSError as e:print('SCRIPT_ACCESS',str(f),type(e).__name__)
 except OSError as e:print('PROCESS_ACCESS',type(e).__name__)
PY
