set -eu
. /root/sdxl_research/env.sh
n=c_mix1_pcm_dm4_matched_time
c="$SDXL_ROOT/tmp/$n/code"
o="$SDXL_ROOT/outputs/$n"
evalcode="$SDXL_ROOT/tmp/bc_coco_winner_20261102/code"
base="$SDXL_ROOT/outputs/bc_winner_late_mix_20261102/late_mix/source"
mkdir -p "$c" "$o"
exec 9>"$o/deploy.lock"
flock -n 9
if test -s "$o/queue.pid" && kill -0 "$(cat "$o/queue.pid")" 2>/dev/null; then echo ALREADY_ACTIVE; exit; fi
cd "$c"
cp "$base/"*.py .
tar -xzf "$SDXL_ROOT/tmp/pcm_dm4_matched_time.tgz"
export PYTHONPATH="$c:$evalcode:$SDXL_ROOT/code/sdxl_research" CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TORCHDYNAMO_DISABLE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
CUDA_VISIBLE_DEVICES='' python -m unittest test_boundary.BoundaryTests.test_terminal_only_dm_keeps_fake_middle_and_terminal test_boundary.BoundaryTests.test_equal_time_and_equal_updates_can_arrive_in_either_order test_boundary.BoundaryTests.test_prefix_is_detached_and_fake_captures_actual_boundary test_paired_stages.PairedStagesTests.test_middle_uses_actual_noisy_boundary test_fake_tracking
python -c 'import run_pcm_dm4_comparison,queue_dm_pair_coco,launch; from pathlib import Path; assert all(Path(m.__file__).resolve().parent==Path.cwd() for m in (run_pcm_dm4_comparison,queue_dm_pair_coco,launch))'
nohup python -u run_pcm_dm4_comparison.py 9>&- >>"$o/queue.log" 2>&1 </dev/null &
echo $! > "$o/queue.pid"
for i in $(seq 1 25); do grep -q VERIFIED_CONFIG "$o/queue.log" && break; kill -0 "$(cat "$o/queue.pid")" 2>/dev/null || break; sleep 1; done
tail -8 "$o/queue.log"
kill -0 "$(cat "$o/queue.pid")"
grep -q VERIFIED_CONFIG "$o/queue.log"
cat "$o/queue.pid"

