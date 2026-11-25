python3 - <<'PY'
from pathlib import Path
import json
b=Path('/data5/shenglin.zeng/checkpoints/a800_archive_20261105')
for p in b.rglob('manifest.json'):print('MANIFEST',p)
for p in b.glob('**/images'):print('IMAGES',p)
for p in b.glob('**/*.tar*'):print('TAR',p)
PY
