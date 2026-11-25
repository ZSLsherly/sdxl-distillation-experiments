set -eu
. /root/sdxl_research/env.sh
cd "$SDXL_ROOT/tmp/A_pcm1_terminal_dm_g160_20261113/code"
export PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python independent_A_terminal_test_n13.py
