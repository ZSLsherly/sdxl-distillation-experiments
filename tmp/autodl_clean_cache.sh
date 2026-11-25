set -eu
/root/sdxl_research/env/bin/python - <<'PY'
from pathlib import Path
import shutil,json,os
before=shutil.disk_usage('/')
paths=[Path('/root/.cache/pip')]+list(Path('/tmp').glob('pip-unpack-*'))+list(Path('/tmp').glob('pip-ephem-wheel-cache-*'))+list(Path('/tmp').glob('pip-install-*'))+list(Path('/tmp').glob('pip-req-tracker-*'))
removed=[]
for p in paths:
 if not p.exists():continue
 assert not p.is_symlink()
 resolved=p.resolve()
 assert resolved==Path('/root/.cache/pip') or (resolved.parent==Path('/tmp') and resolved.name.startswith(('pip-unpack-','pip-ephem-wheel-cache-','pip-install-','pip-req-tracker-')))
 shutil.rmtree(p);removed.append(str(resolved))
os.sync()
after=shutil.disk_usage('/')
r={'removed_cache_directories':removed,'before_free_GiB':before.free/1024**3,'after_free_GiB':after.free/1024**3,'actual_freed_GiB':(after.free-before.free)/1024**3}
Path('/root/sdxl_research/cache_cleanup.json').write_text(json.dumps(r,indent=2));print(json.dumps(r),flush=True)
PY
df -h /
