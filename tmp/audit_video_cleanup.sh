set -e
source /data5/shenglin.zeng/env.sh
du -x -h --max-depth=2 /data5/shenglin.zeng/models /data5/shenglin.zeng/outputs/sparkdiffusion
python - <<'PY'
import json
from pathlib import Path
r=Path('/data5/shenglin.zeng')
for parent in [r/'models',r/'src',r/'cache']:
 print('DIRECTORY',parent)
 for p in sorted(parent.iterdir()):print(p.name,'SYMLINK',p.is_symlink(),'TARGET',str(p.resolve()))
active=json.loads((r/'sdxl_crossdistill/outputs/active_experiment.json').read_text())
for run in active['runs']:
 p=r/'sdxl_crossdistill/outputs'/run['run']
 print('ACTIVE_CONFIG',(p/'config.json').read_text())
 print('ACTIVE_STATUS',(p/'status.json').read_text())
print('VIDEO_FILES')
for p in (r/'outputs/sparkdiffusion').rglob('*'):
 if p.is_file() and p.suffix in ['.yaml','.yml','.json'] and p.stat().st_size<100000:
  print(str(p))
PY
