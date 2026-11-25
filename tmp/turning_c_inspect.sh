set -euo pipefail
source /data5/shenglin.zeng/env.sh
cd /data5/shenglin.zeng/sdxl_crossdistill
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv
df -h .
python - <<'PY'
import json, hashlib
from pathlib import Path
active=Path('outputs/active_experiment.json')
if active.exists():
    m=json.loads(active.read_text())
    for run in m['runs']:
        p=Path('outputs')/run['run']/'status.json'
        print('ACTIVE_RECORD',run)
        if p.exists(): print('STATUS',p.read_text())
for name in ['train.py','batching.py','objectives.py','pixel.py','training_state.py']:
    p=Path('sdxl_parallel')/name
    print('SOURCE_HASH',name,hashlib.sha256(p.read_bytes()).hexdigest())
for p in Path('outputs/gan_stage_joint4_4gpu').glob('*/config.json'):
    print('BASELINE_CONFIG',str(p),p.read_text())
PY
