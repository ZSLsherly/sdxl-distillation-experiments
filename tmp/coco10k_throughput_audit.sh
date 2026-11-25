set -eu
source /data5/shenglin.zeng/env.sh
cd /data5/shenglin.zeng/sdxl_crossdistill
python - <<'PY'
from pathlib import Path
import json
p=Path('outputs/coco10k_g160_20261020')
for rank in range(4):
    latest={}
    for line in (p/f'worker_{rank}.log').read_text().splitlines():
        if line.startswith('PROGRESS '):
            row=json.loads(line[len('PROGRESS '):]);latest[row['method']]=row
        elif 'OOM_REDUCE' in line or 'Traceback' in line:print('WARNING',rank,line)
    print('GPU',rank, json.dumps({m:{k:v[k] for k in ['done','images_per_second','batch','seconds']} for m,v in latest.items()}))
print('PREFLIGHT',json.dumps({k: {'images_per_second':v['images_per_second'],'batch':v['batch']} for k,v in json.loads((p/'validation.json').read_text())['models'].items()}))
print('OUR_WORKERS',(p/'worker_pids.txt').read_text().split())
PY
nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_gpu_memory,process_name --format=csv
nvidia-smi pmon -c 1 -s um
