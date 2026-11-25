set -euo pipefail
cd /data1/shenglin.zeng/sdxl_eval_uniform_20261025
export LD_LIBRARY_PATH="$PWD/runtime/lib:${LD_LIBRARY_PATH:-}"
env/bin/python - <<'PY'
from pathlib import Path
import hashlib,shutil,sys
r=Path.cwd();sys.path.insert(0,str(r/'training_code'))
from launch import gpu_check
print('FREE_GPUS',gpu_check([2,3],36))
assert shutil.disk_usage(r).free>200*2**30
for name in ('evaluate.py','evaluation_core.py','objectives.py'):
 assert hashlib.sha256((r/'training_code'/name).read_bytes()).digest()==hashlib.sha256((r/'code'/name).read_bytes()).digest(),name
assert not (r/'outputs/joint34_uniform_10k_2gpu_20261025/workflow.jsonl').exists()
print('EVALUATOR_IDENTICAL_TO_COMPLETED_10K_RUN')
PY
nohup nice -n 10 ionice -c 2 -n 7 env/bin/python -u joint34_workflow_20261025.py > outputs/joint34_uniform_10k_2gpu_20261025/workflow.log 2>&1 < /dev/null &
echo "$!" > outputs/joint34_uniform_10k_2gpu_20261025/workflow.pid
echo WORKFLOW_STARTED_PID=$!
