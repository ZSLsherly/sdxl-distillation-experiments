set -e
source /data5/shenglin.zeng/env.sh
cd /data5/shenglin.zeng/sdxl_crossdistill
mkdir -p models/dinov2-small
nohup env CUDA_VISIBLE_DEVICES=3 python src/sdxl_prepare_v3.py > logs/gan_v3_prepare.log 2>&1 < /dev/null &
echo PREPARE_PID_$!
