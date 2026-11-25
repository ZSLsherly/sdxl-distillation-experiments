set -eu
d=/data1/shenglin.zeng/sdxl_eval_uniform_20261025/outputs/semantic_references_1024
cd "$d"
export EXCLUDED_UUID=$(nvidia-smi -i 4 --query-gpu=uuid --format=csv,noheader)
../../runtime/bin/python - <<'PY'
import os,signal,time
from pathlib import Path
d=Path.cwd()
exclude=os.environ['EXCLUDED_UUID']
def targets():
 found=[]
 for p in Path('/proc').glob('[0-9]*'):
  try:
   if p.stat().st_uid!=os.getuid() or (p/'cwd').resolve()!=d:continue
   argv=(p/'cmdline').read_bytes().split(b'\0')
   if b'run_references.py' in argv:found.append(('manager',int(p.name)))
   elif b'generate.py' in argv and ('CUDA_VISIBLE_DEVICES='+exclude).encode() in (p/'environ').read_bytes().split(b'\0'):found.append(('gpu4',int(p.name)))
  except (OSError,RuntimeError):continue
 return sorted(found,key=lambda z:z[0]!='manager')
for typ,pid in targets():
 print('stopping own',typ,pid,flush=True);os.kill(pid,signal.SIGTERM)
time.sleep(1)
for typ,pid in targets():os.kill(pid,signal.SIGKILL)
p=d/'run_references.py'
s=p.read_text()
needle="if u.strip() not in busy and"
if needle in s:p.write_text(s.replace(needle,"if u.strip() != "+repr(exclude)+" and u.strip() not in busy and"))
assert repr(exclude) in p.read_text()
time.sleep(.5);assert not targets()
print('EXCLUDED',exclude,flush=True)
PY
nvidia-smi -i 4 --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
