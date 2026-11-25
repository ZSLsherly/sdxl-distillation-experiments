source /data5/shenglin.zeng/env.sh
cd /data5/shenglin.zeng/sdxl_crossdistill
python - <<'PY'
from pathlib import Path
import json
p=Path('outputs/microbatch32_20261023')
for name in ['guard_result.json','guard.log','probe/status.json']:
 f=p/name
 if f.exists():print(name,f.read_text()[-3000:])
f=p/'probe/rank0.jsonl'
if f.exists():
 rows=[json.loads(x) for x in f.read_text().splitlines()]
 print('COUNTS',sum(r['phase']=='student_done' for r in rows),sum(r['phase']=='fake_D' for r in rows))
 for row in rows[-2:]:print('RECENT',json.dumps({k:v for k,v in row.items() if k!='original_indices_rank'}))
f=Path('logs/microbatch32_20261023_probe.log')
if f.exists():print('TRAIN_TAIL',f.read_text()[-3000:])
PY
nvidia-smi -i 0,2 --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv
