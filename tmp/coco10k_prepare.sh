set -eu
source /data5/shenglin.zeng/env.sh
cd /data5/shenglin.zeng/sdxl_crossdistill
python -m zipfile -e coco10k_code.zip .
mkdir -p outputs/coco10k_g160_20261020
nohup bash -c 'CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python -u sdxl_coco10k/evaluate.py prepare > outputs/coco10k_g160_20261020/prepare.log 2>&1; echo $? > outputs/coco10k_g160_20261020/prepare.exit' </dev/null >/dev/null 2>&1 &
echo "PREPARE_PID $!"
