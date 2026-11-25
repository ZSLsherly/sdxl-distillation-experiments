set -euo pipefail
root=/data1/shenglin.zeng/teacher_turning_20261026
cd "$root"
export LD_LIBRARY_PATH="$root/runtime/lib:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TMPDIR="$root/tmp" HF_HOME="$root/cache"
nice -n 10 env/bin/python -u sdxl_turning_batch.py --root "$root" --out "$root/batch_probe" --probe
