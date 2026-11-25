set -eu
source /data5/shenglin.zeng/env.sh
root="$CROSSDISTILL_ROOT/sdxl_crossdistill"
cd "$root"
export CUDA_VISIBLE_DEVICES=0,1,2,3 HF_HUB_OFFLINE=1
nohup bash -c '
root=/data5/shenglin.zeng/sdxl_crossdistill
for i in $(seq 1 360); do
 if [ -f "$root/download_manifest.json" ] && [ -f "$root/datasets/prepared.pt" ]; then
  python "$root/src/sdxl_objectives.py" || exit 1
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
  torchrun --standalone --nnodes=1 --nproc-per-node=4 "$root/src/sdxl_train.py" --updates 20 --resolution 1024 --run smoke_1024
  result=$?
  echo "$result" > "$root/logs/train.exit"
  exit "$result"
 fi
 sleep 10
done
echo "Waiting for downloads/prepared data timed out"
echo 124 > "$root/logs/train.exit"
exit 124
' > logs/train.log 2>&1 < /dev/null &
echo $! > logs/train.pid
cat logs/train.pid
