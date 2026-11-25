source /data5/shenglin.zeng/env.sh
cd /data5/shenglin.zeng/sdxl_crossdistill
python - <<'PY'
from pathlib import Path
import json
p=Path('outputs/joint4_uniform_10k_2gpu_20261023')
for f in [p/'guard_result.json',p/'train/status.json']:
 if f.exists(): print(f.name,f.read_text())
for d in (p/'train/training_state').glob('*'):
 print('STATE',str(d),'complete',(d/'complete.json').exists(),
       'GiB',round(sum(f.stat().st_size for f in d.glob('*') if f.is_file())/2**30,2))
log=Path('logs/joint4_uniform_10k_2gpu_20261023_train.log')
if log.exists():print('LOG_TAIL',log.read_text()[-700:])
PY
nvidia-smi -i 0,2 --query-compute-apps=gpu_uuid,pid,used_memory --format=csv
