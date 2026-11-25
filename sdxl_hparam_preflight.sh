source /data5/shenglin.zeng/env.sh
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv
nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv
df -h /data5
ls -l /data5/shenglin.zeng/sdxl_crossdistill/outputs
