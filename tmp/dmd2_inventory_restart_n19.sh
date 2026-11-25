R=/data1/shenglin.zeng/dmd2_matched_10k_20261119
date
for i in 0 1 2 3 4 5 6 7; do nvidia-smi -i "$i" --query-gpu=index,uuid,name,memory.total,memory.free,utilization.gpu --format=csv,noheader 2>&1; done
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader 2>&1
ps -eo pid,args | grep -E '[r]un_guard.py|[t]rain_matched.py' || true
du -h --max-depth=3 "$R/runs" "$R/tmp" "$R/logs" "$R"/failed* 2>/dev/null | tail -n 35
df -h /data1
