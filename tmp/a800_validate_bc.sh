set -eu
. /root/sdxl_research/env.sh
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)"
tar -xzf "$SDXL_ROOT/tmp/masked_bc_code.tar.gz" -C "$SDXL_ROOT/code"
export PYTHONPATH="$SDXL_ROOT/code/sdxl_research"
python -m unittest test_masked_turning test_turning_sampling test_boundary_global_noise test_boundary test_turning_batch64_workflow
python "$SDXL_ROOT/code/sdxl_research/train.py" --help | grep checkpoint-mode
