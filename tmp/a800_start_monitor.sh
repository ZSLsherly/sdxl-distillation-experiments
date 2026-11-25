set -eu
. /root/sdxl_research/env.sh
nohup python -u "$SDXL_ROOT/tmp/a800_monitor_compare.py" > "$SDXL_WORK/logs/a800_resource_monitor.log" 2>&1 < /dev/null &
echo MONITOR_PID=$!
tail -n 2 "$SDXL_WORK/logs/a800_offload_compare.log"
