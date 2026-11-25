cd /data5/shenglin.zeng/a50k_20261124
python3 - <<'PY'
from pathlib import Path
import os,signal
p=Path('/proc/1623446')
if p.exists() and p.stat().st_uid==os.getuid() and b"root=Path('/data5/shenglin.zeng/a50k_20261124')" in (p/'cmdline').read_bytes() and b"archive.lock" in (p/'cmdline').read_bytes():
 assert not Path('outputs/A50K_G320/status.json').exists()
 os.kill(int(p.name),signal.SIGTERM);print('Replaced waiting archive worker')
PY
