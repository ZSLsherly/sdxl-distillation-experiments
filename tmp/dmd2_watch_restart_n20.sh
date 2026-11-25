R=/data1/shenglin.zeng/dmd2_matched_10k_20261119
date
tail -n 5 "$R/logs/queue_restart_20261120.log"
grep -E 'UPDATE|Error|INIT_RANK_' "$R/logs/formal_g160_restart_20261120.log" | tail -n 10
[ ! -f "$R/runs/formal_g160_restart_20261120/status.json" ] || cat "$R/runs/formal_g160_restart_20261120/status.json"
nvidia-smi --id=0,1,5,6 --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
