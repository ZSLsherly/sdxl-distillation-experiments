set -euo pipefail
source /data5/shenglin.zeng/env.sh
out=/data5/shenglin.zeng/sdxl_crossdistill/outputs/boundary_dmd_10k_a100_20261025_v1
if kill -0 "$(cat "$out/workflow.pid")" 2>/dev/null; then
  state=$(ps -o stat= -p "$(cat "$out/workflow.pid")")
  case "$state" in Z*) ;; *) echo 'Prior workflow is still alive; refuse duplicate'; exit 1;; esac
fi
nohup nice -n 10 python -u "$out/code/run_workflow.py" >> "$out/workflow_retry1.log" 2>&1 < /dev/null &
echo $! > "$out/workflow.pid"
cat "$out/workflow.pid"
