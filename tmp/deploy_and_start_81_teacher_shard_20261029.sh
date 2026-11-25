set -euo pipefail
root=/data3/shenglin.zeng/sdxl_turning_ablation_20261028
cd "$root"
if pgrep -u "$(id -u)" -f '/code/sdxl_research/[t]rain.py --root /data3/shenglin.zeng/sdxl_turning_ablation_20261028';then
 echo EXISTING_TRAINING_DO_NOT_REPLACE;exit 1
fi
tar -xzf tmp/teacher_shard_code.tar.gz -C code
export LD_LIBRARY_PATH="$root/runtime/lib:${LD_LIBRARY_PATH:-}"
export TMPDIR="$root/tmp"
CUDA_VISIBLE_DEVICES='' env/bin/python code/sdxl_research/train.py --help >/dev/null
CUDA_VISIBLE_DEVICES='' env/bin/python code/sdxl_research/test_turning_batch64_workflow.py
test ! -e pipeline_teacher_shard_20261029.log
nohup bash tmp/run_81_pipeline_teacher_shard_20261029.sh > pipeline_teacher_shard_20261029.log 2>&1 < /dev/null &
echo PIPELINE_TEACHER_SHARD_PID=$!
