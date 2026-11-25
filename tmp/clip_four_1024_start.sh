set -eu
. /root/sdxl_research/env.sh
out="$SDXL_WORK/semantic_four_1024_20261103"
mkdir -p "$out"
export PYTHONPATH="$SDXL_ROOT/tmp/bc_coco_winner_20261102/code"
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader
if test -s "$out/worker.pid" && kill -0 "$(cat "$out/worker.pid")" 2>/dev/null; then echo ALREADY_ACTIVE; exit 0; fi
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)"
nohup python -u "$SDXL_ROOT/tmp/clip_four_1024.py" >>"$out/worker.log" 2>&1 </dev/null &
echo $! > "$out/worker.pid"
echo JOB "$(cat "$out/worker.pid")"
