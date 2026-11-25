set -e
source /data5/shenglin.zeng/env.sh
cd /data5/shenglin.zeng/sdxl_crossdistill
python - <<'PY'
import json
from pathlib import Path
m=json.loads(Path('outputs/active_experiment.json').read_text())
for r in m['runs']:
 p=Path('outputs')/r['run'];status=json.loads((p/'status.json').read_text())
 if status['status']=='running':
  (p/'STOP_REQUEST').write_text('User requested switch to DMD-only control.\n')
  print('STOP_REQUESTED',r['run'])
 else:print('ALREADY_INACTIVE',r['run'],status['status'])
PY
df -h /data5/shenglin.zeng
