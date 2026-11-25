set -eu
source /data5/shenglin.zeng/env.sh
root="$CROSSDISTILL_ROOT/sdxl_crossdistill"
test "$(cat "$root/logs/train.exit")" = 0
test -f "$root/models/baselines/manifest.json"
export CUDA_VISIBLE_DEVICES=0,1,2,3 HF_HUB_OFFLINE=1
nohup bash -c '
root=/data5/shenglin.zeng/sdxl_crossdistill
torchrun --standalone --nnodes=1 --nproc-per-node=4 "$root/src/sdxl_compare.py"
result=$?
echo "$result" > "$root/logs/comparison.exit"
exit "$result"
' > "$root/logs/comparison.log" 2>&1 < /dev/null &
echo $! > "$root/logs/comparison.pid"
cat "$root/logs/comparison.pid"
