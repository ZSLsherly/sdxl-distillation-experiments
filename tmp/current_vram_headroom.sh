source /data5/shenglin.zeng/env.sh
cd /data5/shenglin.zeng/sdxl_crossdistill
nvidia-smi --query-gpu=index,memory.total,memory.used,memory.free --format=csv,noheader
python - <<'PY'
import json
from pathlib import Path
m=json.loads(Path('outputs/active_experiment.json').read_text());p=Path('outputs')/m['runs'][0]['run']
print((p/'status.json').read_text())
f=Path(m['runs'][0]['log'])
lines=f.read_text(errors='replace').splitlines()
print('ERRORS',[s for s in lines if 'out of memory' in s.lower() or 'Traceback' in s][-5:])
PY
