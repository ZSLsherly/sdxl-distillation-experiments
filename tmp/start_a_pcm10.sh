set -eu
. /root/sdxl_research/env.sh
c="$SDXL_ROOT/tmp/A_pcm10_g160/code"; o="$SDXL_ROOT/outputs/A_pcm10_g160"
mkdir -p "$c" "$o"
exec 9>"$o/deploy.lock"; flock -n 9
if test -s "$o/queue.pid" && kill -0 "$(cat "$o/queue.pid")" 2>/dev/null;then echo ALREADY_ACTIVE;exit 0;fi
cd "$c"
cp "$SDXL_ROOT/outputs/turning_BC_paired_quota_20261102/uniform_control/source/"*.py .
cp "$SDXL_ROOT/code/sdxl_research/launch.py" .
cp "$SDXL_ROOT/tmp/c_mix1_pcm_dm4_matched_time/code/queue_dm_pair_coco.py" .
cp "$SDXL_ROOT/tmp/run_a_pcm10.py" .
export PYTHONPATH="$c:$SDXL_ROOT/code/sdxl_research:$SDXL_ROOT/tmp/bc_coco_winner_20261102/code" CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TORCHDYNAMO_DISABLE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
nohup python -u run_a_pcm10.py 9>&- >>"$o/queue.log" 2>&1 </dev/null &
echo $! >"$o/queue.pid"
sleep 5
tail -12 "$o/queue.log"; kill -0 "$(cat "$o/queue.pid")";cat "$o/queue.pid"

