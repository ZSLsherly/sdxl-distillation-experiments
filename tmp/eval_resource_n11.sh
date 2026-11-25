date -Iseconds
hostname
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv,noheader
df -h /data1 /data5 2>/dev/null || true
find /data1/shenglin.zeng -maxdepth 3 -type d 2>/dev/null | head -35
