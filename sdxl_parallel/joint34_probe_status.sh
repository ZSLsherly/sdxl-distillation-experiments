source /data5/shenglin.zeng/env.sh
cd /data5/shenglin.zeng/sdxl_crossdistill
python - <<'PY'
import json
from pathlib import Path
m=json.loads(Path('outputs/gan_stage_joint34/probe_launch.json').read_text())
print('LAUNCH',json.dumps(m))
for f in [Path(m['exit']),Path('outputs')/m['run']/'status.json',Path('outputs')/m['run']/'benchmark.json']:
 if f.exists():print(str(f),f.read_text())
log=Path(m['log'])
if log.exists():
 for line in log.read_text(errors='replace').splitlines()[-12:]:print(line[:600])
PY
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
