set -e
source /data5/shenglin.zeng/env.sh
cd /data5/shenglin.zeng/sdxl_crossdistill
python - <<'PY'
import json
from pathlib import Path
r=json.loads(Path('outputs/gan_stage_v3/launch.json').read_text())
for v in ['joint4','gan4']:
 p=Path('outputs')/r[v]['run'];f=p/'config.json'
 if f.exists():
  c=json.loads(f.read_text());assert c['student_lr']==c['fake_lr']==1e-6
  print(v,{'student_lr':c['student_lr'],'fake_lr':c['fake_lr'],'D_lr':c['D_lr'],'batch':c['effective_batch'],'run':str(p)})
 else:print(v,'MODEL_LOADING')
 log=Path('logs')/('gan_v3_'+p.name+'.log')
 if log.exists():print('\n'.join(log.read_text(errors='replace').splitlines()[-4:]))
PY
