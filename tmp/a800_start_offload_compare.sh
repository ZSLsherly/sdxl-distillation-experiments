set -eu
. /root/sdxl_research/env.sh
nohup python -u "$SDXL_ROOT/tmp/a800_offload_compare.py" > "$SDXL_WORK/logs/a800_offload_compare.log" 2>&1 < /dev/null &
echo COMPARE_PID=$!
