R=/data5/shenglin.zeng
df -h / /dev/shm "$R"
free -h
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
ls -ld "$R"/sdxl_crossdistill/models/sdxl-base-1.0 "$R"/envs/*/bin/python "$R"/sdxl_crossdistill/env/bin/python 2>/dev/null || true
ls -lh "$R"/a50k_20261124/datasets/parallel_50k_seed20261018
