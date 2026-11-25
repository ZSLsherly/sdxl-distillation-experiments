date
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.free,utilization.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader
ps -eo pid,args | grep -E '[r]un_guard.py|[t]rain_matched.py' || true
