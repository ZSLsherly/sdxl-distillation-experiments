date -Is
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader,nounits
nvidia-smi --query-gpu=index,utilization.gpu,memory.free --format=csv,noheader,nounits -l 2 | head -n 40
