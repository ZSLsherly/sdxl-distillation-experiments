set -euo pipefail
root=/data3/shenglin.zeng/sdxl_turning_ablation_20261028
cd "$root"
test ! -e pipeline_compat_20261029.log
nohup bash tmp/run_81_pipeline_compat_20261029.sh > pipeline_compat_20261029.log 2>&1 < /dev/null &
echo PIPELINE_COMPAT_PID=$!
