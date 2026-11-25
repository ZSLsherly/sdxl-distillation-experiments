source /data5/shenglin.zeng/env.sh
cd /data5/shenglin.zeng/sdxl_crossdistill
python - <<'PY'
from pathlib import Path
import json
p=Path('outputs/gpu_smoke_20261023')
for name in ['guard_result.json','guard.log','default_b8_g4/status.json']:
 q=p/name
 if q.exists():print(name,'\n',q.read_text()[-3500:])
q=p/'default_b8_g4/rank0.jsonl'
if q.exists():
 rows=[json.loads(x) for x in q.read_text().splitlines()]
 gs=[r for r in rows if r['phase']=='student_done']
 print('COUNTS',len(gs),sum(r['phase']=='fake_D' for r in rows))
 for r in gs:print('G',json.dumps({k:r[k] for k in ['g','stage','cycle_seconds','dmd_loss','gan_loss','pcm_loss','grad_norm','peak_allocated_GiB','peak_reserved_GiB']}))
log=Path('logs/gpu_smoke_20261023_default_b8_g4.log')
if log.exists():print('TRAIN_LOG_TAIL',log.read_text()[-2400:])
PY
nvidia-smi -i 0,2 --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv
