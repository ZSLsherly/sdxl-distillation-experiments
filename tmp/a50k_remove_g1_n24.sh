set -eu
cd /data5/shenglin.zeng/a50k_20261124
python3 - <<'PY'
import os,signal,shutil,json,fcntl
from pathlib import Path
r=Path.cwd()
with (r/'run.lock').open('a') as f:
 fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
 d=r/'outputs/first_update_mb16'
 assert json.loads((d/'status.json').read_text())['student_updates']==1
 for folder in ['training_state','weights']:
  for p in (d/folder).glob('*_g000001*'):
   assert p.is_dir() and p.resolve().parent==d/folder and not p.is_symlink()
   print('DELETE',p, sum(v.stat().st_size for v in p.glob('*') if v.is_file()))
   shutil.rmtree(p)
 p=Path('/proc/1624329')
 if p.exists() and p.stat().st_uid==os.getuid() and b'archive.lock' in (p/'cmdline').read_bytes() and str(r).encode() in (p/'cmdline').read_bytes():
  os.kill(int(p.name),signal.SIGTERM); print('Stopped superseded archive watcher')
 print('G1_remaining',list(d.glob('*/*_g000001*')))
PY
