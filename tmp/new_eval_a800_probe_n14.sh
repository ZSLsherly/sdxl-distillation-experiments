/root/sdxl_research/env/bin/python - <<'PY'
import json
from pathlib import Path
r=Path('/root/sdxl_research/outputs')
for p in r.glob('*2026111[34]*'):
 s=p/'model/status.json'
 if s.exists():
  d=json.loads(s.read_text());print('STATE',p.name,d)
 for w in p.glob('model/weights/*/student_rank0.pt'):print('WEIGHT',w,w.stat().st_size)
 for v in p.glob('**/scores.json'):print('SCORES',v)
PY

