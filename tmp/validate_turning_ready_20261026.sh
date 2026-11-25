set -euo pipefail
root=/data1/shenglin.zeng/teacher_turning_20261026
cd "$root"
export CUDA_VISIBLE_DEVICES=''
export LD_LIBRARY_PATH="$root/runtime/lib:${LD_LIBRARY_PATH:-}"
env/bin/python -u - <<'PY'
import ast,json,hashlib,time
from pathlib import Path
import torch
d=torch.load('conditions.pt',weights_only=True,map_location='cpu')
assert len(set(d['prompts']))==32
hashes={}
for name in ['sdxl_turning_resample.py','run_guard.py']:
 p=Path(name);ast.parse(p.read_text());hashes[name]=hashlib.sha256(p.read_bytes()).hexdigest()
record={'status':'ready_not_started_gpu_occupied','time':time.time(),'prompts':len(d['prompts']),'unique_prompts':len(set(d['prompts'])),'planned_trajectories':76,'gpu_ids':[0,1],'script_sha256':hashes,'training_modified':False}
Path('readiness.json').write_text(json.dumps(record,indent=2))
print(json.dumps(record,indent=2))
PY
