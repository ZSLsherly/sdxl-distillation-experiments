set -eu
. /root/sdxl_research/env.sh
export PYTHONPATH="$SDXL_ROOT/code/sdxl_research"
python -m unittest test_masked_turning test_turning_sampling test_boundary_global_noise test_boundary test_turning_batch64_workflow
python "$SDXL_ROOT/code/sdxl_research/train.py" --help | grep checkpoint-mode
nohup python -u "$SDXL_ROOT/code/sdxl_research/run_masked_bc.py" > "$SDXL_WORK/logs/masked_bc_queue.log" 2>&1 < /dev/null &
echo QUEUE_PID=$!
