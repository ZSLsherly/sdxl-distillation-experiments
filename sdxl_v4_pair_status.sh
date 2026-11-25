set -e
source /data5/shenglin.zeng/env.sh
cd /data5/shenglin.zeng/sdxl_crossdistill
python - <<'PY'
import json
from pathlib import Path
p=Path('outputs/gan_stage_v4/launch.json')
if p.exists():
 r=json.loads(p.read_text())
 for v in ['joint4','gan4']:
  run=Path('outputs')/r[v]['run'];print(v,r[v])
  s=run/'status.json'
  if s.exists():print(s.read_text())
  log=Path('logs')/('gan_v4_'+run.name+'.log')
  if log.exists():print('\n'.join(log.read_text(errors='replace').splitlines()[-12:]))
  e=log.with_suffix('.exit')
  if e.exists():print('EXIT',e.read_text())
else:print('NOT_LAUNCHED')
PY
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
