python3 - <<'PY'
import os,signal
from pathlib import Path
prefix='\nimport json,sys,subprocess,os,hashlib,urllib.request,time\n'
for p in Path('/proc').iterdir():
 if not p.name.isdigit() or int(p.name)==os.getpid():continue
 try:
  argv=(p/'cmdline').read_bytes().decode().split('\0')
  if '-c' not in argv:continue
  code=argv[argv.index('-c')+1]
  if not code.startswith(prefix) or "a50k_sources_20261123" not in code:continue
  if p.stat().st_uid!=os.getuid():continue
  os.kill(int(p.name),signal.SIGTERM);print('STOPPED_OWN_SLOW_DOWNLOADER',p.name)
 except (OSError,ValueError,IndexError):continue
PY
