python3 - <<'PY'
import tarfile
from pathlib import Path
root=Path('/data5/shenglin.zeng/checkpoints/a800_archive_20261106/evidence')
for p in sorted(root.glob('*.tar')):
 with tarfile.open(p) as t:
  for m in t:
   if m.name.endswith(('manifest.json','config.json')):print(p.name,m.name)
PY
