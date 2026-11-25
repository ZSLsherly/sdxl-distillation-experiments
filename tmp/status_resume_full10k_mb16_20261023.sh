source /data5/shenglin.zeng/env.sh
cd /data5/shenglin.zeng/sdxl_crossdistill
python - <<'PY'
from pathlib import Path
import json
p=Path('outputs/joint4_uniform_10k_2gpu_mb16_20261023')
for name in ['guard_result.json','guard.log','train/status.json']:
 q=p/name
 if q.exists():print(name,'\n',q.read_text()[-2500:])
q=p/'train/config.json'
if q.exists():
 c=json.loads(q.read_text());print('CONFIG',json.dumps({k:c.get(k) for k in ['benchmark','train_count','variant','effective_batch','accumulation','microbatch_per_gpu','world_size','schedule','max_student_updates','hours','save_every_updates','experiment_modules']}))
q=p/'train/rank0.jsonl'
if q.exists():
 rows=[json.loads(x) for x in q.read_text().splitlines()]
 gs=[r for r in rows if r['phase']=='student_done']
 print('COUNTS',len(gs),sum(r['phase']=='fake_D' for r in rows))
 for r in gs[-3:]:print('G',json.dumps({k:r[k] for k in ['g','stage','cycle_seconds','dmd_loss','gan_loss','pcm_loss','grad_norm','peak_allocated_GiB','peak_reserved_GiB']}))
log=Path('logs/joint4_uniform_10k_2gpu_mb16_20261023_train.log')
if log.exists():print('TRAIN_LOG_TAIL',log.read_text()[-1800:])
PY
nvidia-smi -i 0,2 --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv

