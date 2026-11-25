R=/data1/shenglin.zeng/dmd2_matched_10k_20261119
tail -n 8 "$R/logs/queue.log"
for f in "$R"/logs/smoke_micro*.log; do echo "$f"; tail -n 28 "$f"; done
find "$R/runs" -maxdepth 3 -name 'status.json' -exec cat {} \;
nvidia-smi --id=0,1,3,5 --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
