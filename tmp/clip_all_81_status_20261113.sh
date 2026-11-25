cd /data3/shenglin.zeng/clip_missing_20261113
cat queue_5.log
ps -p "$(cat queue_5.pid)" -o pid,etime,args
du -sh models arms 2>/dev/null
find arms -name ready.json -o -name status.json -o -name results.json 2>/dev/null
nvidia-smi -i 5 --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv,noheader
tail -n 5 arms/*/worker.log 2>/dev/null

