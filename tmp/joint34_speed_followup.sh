source /data5/shenglin.zeng/env.sh
cd /data5/shenglin.zeng/sdxl_crossdistill
ps -p 3575376,3574011,3598411 -o user:16,pid,lstart,comm
python - <<'PY'
from pathlib import Path
import json,statistics
p=Path('outputs/gan_stage_joint4_4gpu/joint4_20261019T122237_4gpu_mb16/rank0.jsonl')
gs=[r for s in p.read_text().splitlines() if (r:=json.loads(s))['phase']=='student_done']
for a,b in [(1,12),(100,120),(121,140),(141,148),(149,160)]:
 rs=[r for r in gs if a<=r['g']<=b]
 print('OLD_G_RANGE',a,b,'MEAN',round(statistics.mean(r['cycle_seconds'] for r in rs),2),'TIME',rs[0]['timestamp'],rs[-1]['timestamp'])
p=Path('outputs/gan_stage_joint34_4gpu/joint34_20261019T195324_4gpu_mb16/rank0.jsonl')
gs=[r for s in p.read_text().splitlines() if (r:=json.loads(s))['phase']=='student_done']
print('NEW_MEAN',statistics.mean(r['cycle_seconds'] for r in gs[-12:]),'LATEST_G',gs[-1]['g'])
PY
