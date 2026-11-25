set -eu
source /data5/shenglin.zeng/env.sh
root="$CROSSDISTILL_ROOT/sdxl_crossdistill"
cd "$root"
export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1
nohup python -u src/sdxl_prepare.py > logs/prepare.log 2>&1 < /dev/null &
echo $! > logs/prepare.pid
cat logs/prepare.pid
