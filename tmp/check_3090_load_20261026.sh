set -eu
date -Is
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader,nounits
nvidia-smi --query-gpu=index,uuid,memory.total,memory.used,memory.free --format=csv,noheader,nounits
nvidia-smi --query-gpu=index,utilization.gpu,utilization.memory,memory.used,power.draw --format=csv,noheader,nounits -i 0,1 -l 2 | head -n 22
