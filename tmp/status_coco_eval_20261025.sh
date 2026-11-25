set -euo pipefail
cd /data1/shenglin.zeng/sdxl_eval_uniform_20261025
runtime/bin/python - <<'PY'
import json,time
from pathlib import Path
p=Path('outputs/coco10k')
status={'checked_unix':time.time(),'statuses':{},'workers_complete':[]}
for f in sorted(p.glob('status_*.json')):status['statuses'][f.name]=json.loads(f.read_text())
for f in sorted(p.glob('worker_*_done.json')):status['workers_complete'].append(f.name)
for name in ['scores.json','complete.json']:
 if (p/name).exists():status[name]=json.loads((p/name).read_text())
if (p/'guard.jsonl').exists():status['guard_events']=[json.loads(x) for x in (p/'guard.jsonl').read_text().splitlines()][-8:]
for f in sorted(p.glob('worker_*.log')):status[f.name]=f.read_text(errors='replace').splitlines()[-3:]
print(json.dumps(status,ensure_ascii=False,indent=2))
PY
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
