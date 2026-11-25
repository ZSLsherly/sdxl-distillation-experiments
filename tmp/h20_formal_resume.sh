set -eu
. /root/sdxl_research/env.sh
cd /root/autodl-tmp/h20_C_mix1_formal_20261103
test ! -e h20_fp32sam4_vae2_probe.log
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)"
export PYTHONPATH="$PWD" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONFAULTHANDLER=1
L=/root/autodl-tmp/h20_gan_mb32/blas126/nvidia/cublas/lib
export LD_PRELOAD="$L/libcublasLt.so.12:$L/libcublas.so.12"
mv queue.log precheck_busy.log
nohup python -u run_h20_series.py </dev/null > queue.log 2>&1 & echo QUEUE_PID=$!
