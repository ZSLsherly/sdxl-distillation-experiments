set -euo pipefail
cd /data1/shenglin.zeng/sdxl_eval_uniform_20261025
runtime/bin/python - <<'PY'
import json
from pathlib import Path
p=Path('outputs/coco10k');status={}
fs=list((p/'features/joint4_uniform').glob('*.npz'))
status['feature_blocks']=len(fs);status['committed_samples']=sum(int(f.stem.split('_')[1])-int(f.stem.split('_')[0]) for f in fs)
status['guard_last']=[json.loads(x) for x in (p/'guard.jsonl').read_text().splitlines()][-4:]
for name in ('scores.json','complete.json'):
 if (p/name).exists():status[name]=json.loads((p/name).read_text())
if (p/'metrics.log').exists():status['metrics_tail']=(p/'metrics.log').read_text().splitlines()[-5:]
print(json.dumps(status,indent=2))
PY
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader
