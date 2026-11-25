set -eu
. /root/sdxl_research/env.sh
r="$SDXL_ROOT/outputs/A_terminal_low30_g160_20261113"
cat "$r/queue.pid"
ps -eo pid,ppid,comm,args | awk '$3 ~ /python|torchrun/ && /A_terminal_low30|run_A_terminal_prior/ {print substr($0,1,210)}'
cat "$r/model/status.json"
