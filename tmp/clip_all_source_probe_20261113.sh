python3 - <<'PY'
import json
from pathlib import Path
b=Path('/data5/shenglin.zeng/checkpoints')
for name,root in [('archive05',b/'a800_archive_20261105/outputs'),('archive09',b/'a800_archive_20261109/outputs')]:
 for p in root.glob('*/model/config.json'): print('CONFIG',p)
 for p in root.glob('*/model/weights/*/student_rank0.pt'):print('WEIGHT',p,p.stat().st_size)
for p in (b/'a800_archive_20261106').glob('**/*.tar'):print('TAR',p,p.stat().st_size)
for p in (b/'h20_gan_20261104').rglob('manifest.json'):print('MANIFEST',p)
PY
