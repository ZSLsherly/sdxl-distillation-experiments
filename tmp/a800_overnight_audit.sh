set -eu
. /root/sdxl_research/env.sh
export CUDA_VISIBLE_DEVICES=''
export PYTHONPATH="$SDXL_ROOT/code/sdxl_research"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
python "$SDXL_ROOT/tmp/a800_overnight_audit.py"
python -m unittest test_paired_stages test_matched_prior test_interval_quota test_masked_turning test_turning_sampling test_boundary_global_noise test_boundary test_turning_batch64_workflow
