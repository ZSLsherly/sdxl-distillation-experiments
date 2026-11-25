source /data5/shenglin.zeng/env.sh
root=/data5/shenglin.zeng/sdxl_crossdistill
tail -n 8 "$root/logs/teacher_turning.log"
python - <<'PY'
import json
from pathlib import Path
p=Path('/data5/shenglin.zeng/sdxl_crossdistill/outputs/hparam_pilot/teacher_turning/result.json')
if p.exists():
 r=json.loads(p.read_text())
 for name,field in r['modes']['base_dt10_bf16']['fields'].items():
  print(name,{k:v for k,v in field.items() if k!='bandwidth_sensitivity'})
else:print('SUMMARY_NOT_YET_WRITTEN')
PY
