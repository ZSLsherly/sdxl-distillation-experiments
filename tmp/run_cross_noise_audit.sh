set -eu
. /root/sdxl_research/env.sh
export CUDA_VISIBLE_DEVICES=''
python "$SDXL_ROOT/tmp/audit_cross_noise_math.py"
