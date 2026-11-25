set -eu
. /root/sdxl_research/env.sh
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
cd "$SDXL_WORK/semantic_abc_20261103"
nice -n 10 python -u "$SDXL_ROOT/tmp/abc_semantic_score.py"
