set -euo pipefail
cd /data1/shenglin.zeng/sdxl_eval_uniform_20261025
export LD_LIBRARY_PATH="$PWD/runtime/lib:${LD_LIBRARY_PATH:-}"
CUDA_VISIBLE_DEVICES='' env/bin/python prepare_joint34_four_gpu_20261025.py
nohup nice -n 10 ionice -c 2 -n 7 env/bin/python -u joint34_four_gpu_workflow_20261025.py > outputs/joint34_uniform_10k_4gpu_20261025/workflow.log 2>&1 < /dev/null &
echo "$!" > outputs/joint34_uniform_10k_4gpu_20261025/workflow.pid
echo FOUR_GPU_WORKFLOW_PID=$!
