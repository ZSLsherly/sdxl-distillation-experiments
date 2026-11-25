set -euo pipefail
ROOT=/data1/shenglin.zeng/sdxl_eval_uniform_20261025
export LD_LIBRARY_PATH="$ROOT/runtime/lib:${LD_LIBRARY_PATH:-}"
cd "$ROOT"
out="$ROOT/outputs/turning_boundary_no_offload_1gpu_probe_20261028"
test ! -e "$out"
mkdir -p "$out"
cp "$ROOT/tmp/run_single_no_offload_probe_20261028.py" "$out/workflow.py"
nohup env/bin/python -u "$out/workflow.py" > "$out/workflow.log" 2>&1 < /dev/null &
echo PROBE_PID=$!
