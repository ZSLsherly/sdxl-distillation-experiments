set -euo pipefail
source /data5/shenglin.zeng/env.sh
cd /data5/shenglin.zeng/sdxl_crossdistill
export OMP_NUM_THREADS=4
CUDA_VISIBLE_DEVICES=0 python src/sdxl_original_control.py --idx 113 > logs/original_control_113.log 2>&1 &
p1=$!
CUDA_VISIBLE_DEVICES=3 python src/sdxl_original_control.py --idx 118 > logs/original_control_118.log 2>&1 &
p2=$!
wait "$p1"
wait "$p2"
tar -czf outputs/original_sdxl_control.tar.gz -C outputs original_sdxl_control
echo ORIGINAL_CONTROL_COMPLETE
