set -euo pipefail
root=/data3/shenglin.zeng/sdxl_turning_ablation_20261028
cd "$root"
exec 9>pipeline.lock
flock -n 9
trap 'echo PIPELINE_FAILED_LINE=$LINENO' ERR
export LD_LIBRARY_PATH="$root/runtime/lib:${LD_LIBRARY_PATH:-}"
export TMPDIR="$root/tmp"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo ALLOCATOR="$PYTORCH_CUDA_ALLOC_CONF"
echo PHASE=four_gpu_smoke
out=outputs/smoke_four_gpu_expandable_20261029
test ! -e "$out"
mkdir -p "$out"
cp tmp/run_four_gpu_expandable_20261029.py "$out/workflow.py"
env/bin/python -u "$out/workflow.py" > "$out/workflow.log" 2>&1
echo PHASE=batch64_validation_then_G160
env/bin/python -u code/sdxl_research/turning_batch64_workflow.py --root "$root" --gpus 1,2,4,5 --min-free-gib 22 --name turning_batch64_g160_4gpu_20261029 > outputs/batch64_workflow.log 2>&1
echo PIPELINE_COMPLETE
