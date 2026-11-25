python3 - <<'PY'
from pathlib import Path
import json
r=Path('/data5/shenglin.zeng/checkpoints/a800_archive_20261114')
for p in r.rglob('student_rank0.pt'):print('WEIGHT',p,p.stat().st_size)
for p in r.rglob('COMPLETE.json'):print('RECEIPT',p,p.read_text()[:1200])
PY
command -v rclone
find /data5/shenglin.zeng/tools -maxdepth 4 -name rclone -type f
