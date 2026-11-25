set -euo pipefail
root=/data3/shenglin.zeng/sdxl_turning_ablation_20261028
cd "$root"
test ! -e outputs/smoke_four_gpu_expandable_retry1_20261029
bash -n tmp/run_81_pipeline_expandable_retry1_20261029.sh
nohup bash tmp/run_81_pipeline_expandable_retry1_20261029.sh >> pipeline_expandable_retry1_20261029.log 2>&1 < /dev/null &
echo PIPELINE_PID=$!
