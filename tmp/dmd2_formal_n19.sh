R=/data1/shenglin.zeng/dmd2_matched_10k_20261119
date
tail -n 5 "$R/logs/queue.log"
grep -E 'UPDATE|Error|INIT_RANK_' "$R/logs/formal_g160.log" | tail -n 8
[ ! -f "$R/runs/formal_g160/status.json" ] || cat "$R/runs/formal_g160/status.json"
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader
df -h /data1
