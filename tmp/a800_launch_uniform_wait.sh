set -eu
. /root/sdxl_research/env.sh
tar -xzf "$SDXL_ROOT/tmp/uniform_control.tar.gz" -C "$SDXL_ROOT/code"
export PYTHONPATH="$SDXL_ROOT/code/sdxl_research"
cd "$SDXL_ROOT/code/sdxl_research"
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest test_uniform_control test_paired_stages test_matched_prior
nohup python -u "$SDXL_ROOT/code/sdxl_research/run_uniform_after_bc.py" > "$SDXL_WORK/logs/uniform_after_bc.log" 2>&1 < /dev/null &
echo WAIT_QUEUE_PID=$!
