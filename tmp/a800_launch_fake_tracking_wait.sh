set -eu
. /root/sdxl_research/env.sh
code="$SDXL_ROOT/tmp/fake_tracking_pilot_20261102/code"
mkdir -p "$(dirname "$code")"
mkdir "$code"
cp "$SDXL_ROOT"/code/sdxl_research/*.py "$code/"
cp -r "$SDXL_ROOT/code/sdxl_research/configs" "$SDXL_ROOT/code/sdxl_research/turning_profiles" "$code/"
tar -xzf "$SDXL_ROOT/tmp/fake_tracking_pilot_v1.tar.gz" -C "$code"
cd "$code"
export PYTHONPATH="$code"
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest test_fake_tracking test_paired_stages test_uniform_control test_matched_prior
nohup python -u run_fake_tracking_pilot.py > "$SDXL_WORK/logs/fake_tracking_pilot_queue.log" 2>&1 </dev/null &
echo QUEUE_PID=$!
