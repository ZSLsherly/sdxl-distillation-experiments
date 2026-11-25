set -eu
. /root/sdxl_research/env.sh
d=/root/autodl-tmp/h20_gan_mb32
c="$d/code_rows"
mkdir -p "$c"
cp "$d/code"/*.py "$d/code"/*.json "$c/"
cd "$c"
tar -xzf "$SDXL_WORK/tmp/h20_rows.tar.gz"
mv sam_pixel_rows.py sam_pixel.py
L="$d/blas126/nvidia/cublas/lib"; export LD_PRELOAD="$L/libcublasLt.so.12:$L/libcublas.so.12" PYTHONFAULTHANDLER=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
CUDA_VISIBLE_DEVICES='' python test_rows.py
sed -e 's/h20_gan_mb32_a4/h20_gan_mb32_rows/g' run_h20.py > run_rows.py
nohup python -u run_rows.py > h20.log 2>&1 </dev/null & echo $!
