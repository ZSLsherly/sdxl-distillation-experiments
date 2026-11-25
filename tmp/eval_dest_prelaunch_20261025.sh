set -e
cd /data1/shenglin.zeng/sdxl_eval_uniform_20261025
ls -lh transfers
nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader
