/root/sdxl_research/env/bin/python - <<'PY'
from pathlib import Path
import json
for p in Path('/root/sdxl_research/outputs').glob('*/model/status.json'):
 d=json.loads(p.read_text())
 if d.get('status')=='complete':
  weights=list(p.parent.glob('weights/*/student_rank0.pt'))
  if weights:print(p.parent.parent.name,d.get('student_updates'),[str(w) for w in weights])
PY
