nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader
ps -p 5375 -o pid,etime,args
tail -4 /data3/shenglin.zeng/clip_missing_20261113/queue_5.log
