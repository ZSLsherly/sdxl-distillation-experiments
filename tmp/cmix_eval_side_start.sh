set -eu
. /root/sdxl_research/env.sh
d="$SDXL_ROOT/tmp/cmix_eval_side_20261103";p="$SDXL_ROOT/outputs/bc_winner_late_mix_20261102/coco10k_C_mix1"
mkdir -p "$d" "$p"
if test -f "$p/queue.pid" && kill -0 "$(cat "$p/queue.pid")" 2>/dev/null; then echo ALREADY_ACTIVE; exit 0; fi
tar -xzf "$SDXL_ROOT/tmp/cmix_eval_side.tar.gz" -C "$d"
cd "$d"
export PYTHONPATH="$SDXL_ROOT/tmp/bc_coco_winner_20261102/code" CUDA_VISIBLE_DEVICES=''
nohup python -u cmix_run_eval_side.py >>"$p/queue.log" 2>&1 </dev/null &
echo $! >"$p/queue.pid"; echo QUEUE_PID=$!
