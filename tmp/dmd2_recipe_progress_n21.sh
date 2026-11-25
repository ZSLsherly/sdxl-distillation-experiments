R=/data1/shenglin.zeng/dmd2_official_recipe_20261121
date
tail -n 4 "$R/logs/queue.log" 2>/dev/null || true
for f in "$R"/logs/smoke_micro*.log "$R"/logs/formal_g160.log; do
  if [ -f "$f" ]; then echo "$f"; tail -n 3 "$f"; fi
done
for f in "$R"/runs/*/status.json; do if [ -f "$f" ]; then echo "$f"; cat "$f"; fi; done
nvidia-smi -i 3,7 --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
df -h /data1
