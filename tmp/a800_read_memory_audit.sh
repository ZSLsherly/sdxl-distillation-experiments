. /root/sdxl_research/env.sh
python - <<'PY'
from pathlib import Path
import json
p=Path('/root/sdxl_research/outputs/a800_memory_audit_20261101/rank0.jsonl')
if p.exists():
 for line in p.read_text().splitlines():
  r=json.loads(line)
  if r['phase']=='memory_audit':print(json.dumps(r))
PY
tail -n 7 /root/sdxl_research/logs/a800_memory_audit_20261101.log
