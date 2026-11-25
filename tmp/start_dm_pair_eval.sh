set -eu
. /root/sdxl_research/env.sh
c="$SDXL_ROOT/tmp/bc_coco_winner_20261102/code"
o="$SDXL_ROOT/outputs/c_mix1_dm_pair_coco10k"
mkdir -p "$o"
cd "$c"
export PYTHONPATH="$c:$SDXL_ROOT/code/sdxl_research" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
if test -f "$o/queue.pid" && kill -0 "$(cat "$o/queue.pid")" 2>/dev/null;then echo ALREADY_ACTIVE; exit 0;fi
python -m py_compile "$SDXL_ROOT/tmp/queue_dm_pair_coco.py"
python -c 'from run_bc_coco_then_winner import check_stop;check_stop()'
nohup python -u "$SDXL_ROOT/tmp/queue_dm_pair_coco.py" >>"$o/queue.log" 2>&1 </dev/null &
echo $! >"$o/queue.pid"
for i in $(seq 1 15); do test -f "$o/baseline.json" && break; kill -0 "$(cat "$o/queue.pid")" 2>/dev/null || break; sleep 1; done
tail -15 "$o/queue.log"
kill -0 "$(cat "$o/queue.pid")"
test -s "$o/baseline.json"
cat "$o/queue.pid"

