set -euo pipefail
cd /data1/shenglin.zeng/sdxl_eval_uniform_20261025
export LD_LIBRARY_PATH="$PWD/runtime/lib:${LD_LIBRARY_PATH:-}"
cp /data1/shenglin.zeng/run_turning_no_offload_4090_20261027.py outputs/turning_boundary_no_offload_2gpu_20261027/workflow_raw.py
nohup env/bin/python -u outputs/turning_boundary_no_offload_2gpu_20261027/workflow_raw.py raw_angle >> outputs/turning_boundary_no_offload_2gpu_20261027/workflow.log 2>&1 < /dev/null &
echo RAW_WORKFLOW_PID=$!
