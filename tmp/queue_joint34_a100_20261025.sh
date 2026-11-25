set -euo pipefail
source /data5/shenglin.zeng/env.sh
cd /data5/shenglin.zeng/sdxl_crossdistill
test ! -f outputs/joint34_uniform_10k_2gpu_20261025/workflow.jsonl
nohup nice -n 10 ionice -c 2 -n 7 python -u joint34_a100_workflow_20261025.py > outputs/joint34_uniform_10k_2gpu_20261025/workflow.log 2>&1 < /dev/null &
echo "$!" > outputs/joint34_uniform_10k_2gpu_20261025/workflow.pid
echo QUEUED_WORKFLOW_PID=$!
