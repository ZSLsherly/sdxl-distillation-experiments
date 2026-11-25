set -eu
. /root/sdxl_research/env.sh
c="$SDXL_ROOT/tmp/interval_D_20261103/code"
o="$SDXL_ROOT/outputs/interval_D_20261103"
mkdir -p "$c" "$o"
if test -f "$o/queue.pid" && kill -0 "$(cat "$o/queue.pid")" 2>/dev/null; then echo ALREADY_RUNNING; exit 0; fi
test ! -e "$o/D" && test ! -e "$o/D.log"
src="$SDXL_ROOT/tmp/bc_coco_winner_20261102/code"
cp "$src"/*.py "$c"/
cp -r "$src/configs" "$src/turning_profiles" "$c"/
tar -xzf "$SDXL_ROOT/tmp/interval_D_changes.tar.gz" -C "$c"
cp "$SDXL_ROOT/tmp/run_interval_D.py" "$c/"
cd "$c"
export PYTHONPATH="$c"
CUDA_VISIBLE_DEVICES='' python -m unittest test_interval_quota test_paired_stages
nvidia-smi --query-gpu=name,memory.free,utilization.gpu --format=csv,noheader
nohup python -u run_interval_D.py </dev/null >"$o/queue.log" 2>&1 &
echo $! > "$o/queue.pid"
sleep 4
cat "$o/queue.log"
kill -0 "$(cat "$o/queue.pid")"
