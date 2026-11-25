set -eu
. /root/sdxl_research/env.sh
name=C_mix1_pcm_dm4_fake4_low30_g160
c="$SDXL_ROOT/tmp/$name/code"
o="$SDXL_ROOT/outputs/$name"
python "$SDXL_ROOT/tmp/prepare_terminal_low30.py"
cp "$SDXL_ROOT/tmp/test_terminal_low30.py" "$SDXL_ROOT/tmp/run_terminal_low30.py" "$c/"
cd "$c"
export PYTHONPATH="$c:$SDXL_ROOT/tmp/bc_coco_winner_20261102/code:$SDXL_ROOT/code/sdxl_research" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TORCHDYNAMO_DISABLE=1
CUDA_VISIBLE_DEVICES='' python test_terminal_low30.py
export CUDA_VISIBLE_DEVICES=0
exec 9>"$o/deploy.lock"
flock -n 9
if test -s "$o/queue.pid" && kill -0 "$(cat "$o/queue.pid")" 2>/dev/null; then echo ALREADY_ACTIVE; exit; fi
nohup python -u run_terminal_low30.py 9>&- >>"$o/queue.log" 2>&1 </dev/null &
echo $! > "$o/queue.pid"
sleep 3
kill -0 "$(cat "$o/queue.pid")"
cat "$o/queue.log"
cat "$o/cpu_validation.json"
