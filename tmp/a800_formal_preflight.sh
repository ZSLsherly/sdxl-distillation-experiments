set -eu
. /root/sdxl_research/env.sh
date -Is
df -h / /root/autodl-tmp
nvidia-smi --query-gpu=index,name,memory.free,utilization.gpu,uuid --format=csv,noheader,nounits
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_gpu_memory --format=csv,noheader,nounits
