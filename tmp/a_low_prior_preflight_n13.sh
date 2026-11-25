set -eu
date -Iseconds
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv,noheader
df -h /root /root/autodl-tmp
ls /root/sdxl_research/env/bin/python /root/sdxl_research/outputs/turning_BC_paired_quota_20261102/uniform_control/config.json
