set -eu
. /root/sdxl_research/env.sh
d=/root/autodl-tmp/h20_C_mix1_formal_20261103
mkdir "$d"
cd "$d"
cp /root/autodl-tmp/h20_gan_mb16/code_fp32sam4/*.py /root/autodl-tmp/h20_gan_mb16/code_fp32sam4/config.json .
tar -xf /root/autodl-tmp/h20_C_formal_scripts.tar
export PYTHONPATH="$d" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONFAULTHANDLER=1
L=/root/autodl-tmp/h20_gan_mb32/blas126/nvidia/cublas/lib
export LD_PRELOAD="$L/libcublasLt.so.12:$L/libcublas.so.12"
python add_eval.py
python -m py_compile train.py run_h20_series.py vae_compare.py
nohup python -u run_h20_series.py </dev/null > queue.log 2>&1 & echo QUEUE_PID=$!

