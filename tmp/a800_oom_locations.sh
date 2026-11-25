. /root/sdxl_research/env.sh
python - <<'PY'
import json
from pathlib import Path
p=Path('/root/autodl-tmp/sdxl_runs/outputs/a800_B_batch_probe_20261101')
for r in json.loads((p/'results.json').read_text()):
 print('BATCH',r['micro_batch'])
 lines=r.get('tail','').splitlines()
 for line in lines:
  if 'train.py' in line or 'in main' in line or 'OutOfMemory' in line or 'backward' in line or 'predict(' in line or 'generate_fake' in line:print(line)
PY
