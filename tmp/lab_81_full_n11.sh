python3 - <<'PY'
from pathlib import Path
import json
r=Path('/data3/shenglin.zeng/sdxl_turning_ablation_20261028')
for p in r.glob('outputs/**/status.json'):
 if 'g160' not in str(p):continue
 s=json.loads(p.read_text());print(p, {k:s.get(k) for k in ['status','student_updates','fake_updates']})
 if s.get('student_updates')==160:
  print('WEIGHTS',[(str(q),q.stat().st_size) for q in p.parent.glob('weights/*/student_rank*.pt')][-8:])
PY
