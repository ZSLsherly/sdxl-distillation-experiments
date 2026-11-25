R=/data1/shenglin.zeng/dmd2_matched_10k_20261119
tail -n 5 "$R/logs/queue.log"
grep -E 'UPDATE|Error|INIT_RANK_' "$R/logs/smoke_micro16.log" | tail -n 8
for f in "$R"/runs/*/status.json "$R"/smoke_passed.json; do [ ! -f "$f" ] || cat "$f"; done
nvidia-smi --id=0,1,3,5 --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
