set -e
source /data5/shenglin.zeng/env.sh
cd /data5/shenglin.zeng/sdxl_crossdistill
tar -xzf src/benchmark_vendor.tar.gz -C src
mkdir -p outputs/benchmark_curves_20261018
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2 python src/sdxl_benchmark_curves.py
