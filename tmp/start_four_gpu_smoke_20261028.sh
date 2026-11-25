set -euo pipefail
root=/data3/shenglin.zeng/sdxl_turning_ablation_20261028
export LD_LIBRARY_PATH="$root/runtime/lib:${LD_LIBRARY_PATH:-}"
cd "$root"
out="$root/outputs/smoke_four_gpu_20261028"
test ! -e "$out"
mkdir -p "$out"
cp "$root/tmp/run_four_gpu_smoke_20261028.py" "$out/workflow.py"
nohup env/bin/python -u "$out/workflow.py" > "$out/workflow.log" 2>&1 < /dev/null &
echo SMOKE_PID=$!
