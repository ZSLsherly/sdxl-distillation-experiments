set -eu
. /root/sdxl_research/env.sh
export CUDA_VISIBLE_DEVICES=''
export PYTHONPATH="$SDXL_ROOT/code/sdxl_research"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
python "$SDXL_ROOT/tmp/a800_overnight_audit.py" B
python "$SDXL_ROOT/tmp/a800_overnight_audit.py" C
python "$SDXL_ROOT/tmp/a800_check_final_cpu.py" B
df -h /root/autodl-tmp
