set -euo pipefail
cd /data1/shenglin.zeng/sdxl_eval_uniform_20261025
export LD_LIBRARY_PATH="$PWD/runtime/lib:${LD_LIBRARY_PATH:-}"
nohup runtime/bin/python -u guard_coco_eval_20261025.py > outputs/coco10k/guard_launch.log 2>&1 < /dev/null &
echo GUARD_PID=$!
