set -eu
source /data5/shenglin.zeng/env.sh
cd /data5/shenglin.zeng/sdxl_crossdistill
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 python sdxl_coco10k/snapshot.py
