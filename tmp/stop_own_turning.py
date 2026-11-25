import os,signal
from pathlib import Path
owner=os.getuid()
launcher=3049986
try:
 p=Path(f'/proc/{launcher}')
 args=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
 assert p.stat().st_uid==owner and 'teacher_turning' in args
 os.kill(launcher,signal.SIGTERM);print('STOPPED_OWN_TURNING_LAUNCHER',launcher)
except FileNotFoundError:pass
for p in Path('/proc').glob('[0-9]*'):
 try:
  if p.stat().st_uid!=owner:continue
  args=[x.decode(errors='replace') for x in (p/'cmdline').read_bytes().split(b'\0') if x]
  if not args:continue
  if any(x in ['src/sdxl_teacher_turning.py','/data5/shenglin.zeng/sdxl_crossdistill/src/sdxl_teacher_turning.py'] for x in args) and ('python' in Path(args[0]).name or 'torchrun' in Path(args[0]).name):
   print('STOPPED_OWN_TURNING_PROCESS',p.name)
   os.kill(int(p.name),signal.SIGTERM)
 except (ProcessLookupError,FileNotFoundError,PermissionError):pass
marker=Path('/data5/shenglin.zeng/sdxl_crossdistill/logs/teacher_turning.exit')
if marker.exists():
 print('REMOVED_OWN_FAILED_MARKER',marker.read_text().strip());marker.unlink()
