python3 - <<'PY'
from pathlib import Path
import os
for p in Path('/proc').glob('[0-9]*'):
 try:
  exe=Path(os.readlink(p/'exe')).name
  if 'discovery-ctl' not in exe or 'upload-file' not in (p/'cmdline').read_bytes().decode(errors='replace'): continue
  print('upload PID',p.name,exe,(p/'io').read_text().splitlines()[0])
  for fd in (p/'fd').iterdir():
   if Path(os.readlink(fd)).name=='prepared.pt':
    print('prepared.pt', (p/'fdinfo'/fd.name).read_text().splitlines()[0])
 except (OSError, PermissionError): pass
PY
