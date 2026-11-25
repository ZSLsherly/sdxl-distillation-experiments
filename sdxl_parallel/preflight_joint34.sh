set -e
source /data5/shenglin.zeng/env.sh
cd /data5/shenglin.zeng/sdxl_crossdistill
sha256sum sdxl_parallel/train.py sdxl_parallel/batching.py sdxl_parallel/pixel.py sdxl_parallel/objectives.py
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
df -h /data5/shenglin.zeng
free -h
find outputs/gan_stage_parallel -maxdepth 2 -name STOP_REQUEST -print
