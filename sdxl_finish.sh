set -eu
source /data5/shenglin.zeng/env.sh
root="$CROSSDISTILL_ROOT/sdxl_crossdistill"
cd "$root"
nohup bash -c '
set -eu
root=/data5/shenglin.zeng/sdxl_crossdistill
for i in $(seq 1 540); do
 if [ -f "$root/logs/train.exit" ]; then
  result=$(cat "$root/logs/train.exit")
  if [ "$result" != 0 ]; then echo "Training failed: $result"; exit "$result"; fi
  python -u "$root/src/sdxl_export.py"
  for j in $(seq 1 540); do
   if [ -f "$root/models/baselines/manifest.json" ] && [ -f "$root/datasets/dmd2/coco10k.zip" ]; then
    CUDA_VISIBLE_DEVICES=0,1,2,3 HF_HUB_OFFLINE=1 torchrun --standalone --nnodes=1 --nproc-per-node=4 "$root/src/sdxl_compare.py"
    echo 0 > "$root/logs/comparison.exit"
    exit 0
   fi
   sleep 10
  done
  echo "Waiting for comparison checkpoints timed out";exit 124
 fi
 sleep 10
done
echo "Waiting for training timed out";exit 124
' > logs/finish.log 2>&1 < /dev/null &
echo $! > logs/finish.pid
cat logs/finish.pid
