set -euo pipefail
root=/data3/shenglin.zeng/sdxl_turning_ablation_20261028
cd "$root"
test ! -e pipeline_retry_20261029.log
nohup bash tmp/run_81_pipeline_retry_20261029.sh > pipeline_retry_20261029.log 2>&1 < /dev/null &
echo PIPELINE_RETRY_PID=$!
