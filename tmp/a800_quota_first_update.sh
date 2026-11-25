. /root/sdxl_research/env.sh
python - <<'PY'
import json
from pathlib import Path
p=Path('/root/sdxl_research/outputs/turning_BC_interval_quota_20261101/B/rank0.jsonl')
rows=[json.loads(s) for s in p.read_text().splitlines()]
gs=[r for r in rows if r['phase']=='student_done']
print(json.dumps(gs[-1] if gs else {'completed_student_updates':0}))
PY
