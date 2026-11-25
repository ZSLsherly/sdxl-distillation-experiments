set -eu
. /root/sdxl_research/env.sh
n=A_pcm10_terminal_low30_g160_20261113
c="$SDXL_ROOT/tmp/$n/code"
o="$SDXL_ROOT/outputs/$n"
python "$SDXL_ROOT/tmp/prepare_A_terminal_prior_n13.py"
cp "$SDXL_ROOT/tmp/check_A_terminal_prior_n13.py" "$SDXL_ROOT/tmp/run_A_terminal_prior_n13.py" "$c/"
cd "$c"
export PYTHONPATH="$c:$SDXL_ROOT/tmp/bc_coco_winner_20261102/code:$SDXL_ROOT/code/sdxl_research" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TORCHDYNAMO_DISABLE=1
CUDA_VISIBLE_DEVICES='' python check_A_terminal_prior_n13.py
export CUDA_VISIBLE_DEVICES=0
exec 9>"$o/deploy.lock"
flock -n 9
if test -s "$o/queue.pid" && kill -0 "$(cat "$o/queue.pid")" 2>/dev/null; then echo ALREADY_ACTIVE; exit; fi
nohup python -u run_A_terminal_prior_n13.py 9>&- >>"$o/queue.log" 2>&1 </dev/null &
echo $! >"$o/queue.pid"
sleep 3
kill -0 "$(cat "$o/queue.pid")"
cat "$o/cpu_validation.json"
tail -5 "$o/queue.log"
