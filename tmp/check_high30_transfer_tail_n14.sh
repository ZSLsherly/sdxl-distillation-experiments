python3 - <<'PY'
from pathlib import Path
import os,json
name='A_clean_global_high30_g160_20261114_v2'
root=Path('/data5/shenglin.zeng/checkpoints/a800_archive_20261114/outputs')/name/'model/weights/20261114T172912+0800_g000160_end'
size=5135506134
for i in range(16):
 p=root/f'student_rank0.pt.fzrange16_{i}';start,end=size*i//16,size*(i+1)//16
 if p.exists() and p.stat().st_size<end:print('REMAINING',i,end-p.stat().st_size,'bytes')
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  cmd=(p/'cmdline').read_bytes().split(b'\0')
  if cmd[0].endswith(b'/fzsftp'):
   status=(p/'status').read_text();ppid=next(x.split()[1] for x in status.splitlines() if x.startswith('PPid:'))
   parent=(Path('/proc')/ppid/'cmdline').read_bytes()
   if ("GROUP='"+name+"'").encode() not in parent:continue
   files=[]
   for fd in (p/'fd').iterdir():
    try:
     target=os.readlink(fd)
     if name in target:files.append(target)
    except OSError:pass
   print('OWN_FZSFTP',p.name,'parent',ppid,'files',json.dumps(files))
 except (OSError,IndexError,StopIteration):pass
PY
