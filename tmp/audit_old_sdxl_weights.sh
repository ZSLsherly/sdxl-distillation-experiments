set -e
source /data5/shenglin.zeng/env.sh
python - <<'PY'
import json
from pathlib import Path
r=Path('/data5/shenglin.zeng/sdxl_crossdistill/outputs')
names=['hparam_pilot','gan_stage_pilot','gan_stage_pilot_v2','gan_stage_v3','gan_stage_v4','smoke_1024','v4_validation','state_tests']
for name in names:
 p=r/name
 print('GROUP',name)
 for f in sorted(p.rglob('*')):
  if f.is_file() and (f.suffix in ['.pt','.pth','.safetensors','.ckpt','.bin','.distcp'] or f.stat().st_size>100*1024**2):
   print(json.dumps({'path':str(f),'bytes':f.stat().st_size,'symlink':f.is_symlink()}))
print('ACTIVE',(r/'active_experiment.json').read_text())
PY
