source /data5/shenglin.zeng/env.sh
cd /data5/shenglin.zeng/sdxl_crossdistill
date -Is
df -h / /data5
nvidia-smi --query-gpu=index,memory.free,memory.used,utilization.gpu --format=csv
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv
ps -eo pid,etime,args | grep -E 'sdxl_parallel/train|sdxl_turning|coco10k|torchrun' | grep -v grep
cat outputs/active_experiment.json
cat outputs/gan_stage_joint4_4gpu/joint4_20261019T122237_4gpu_mb16/config.json
sha256sum sdxl_parallel/{train,batching,pixel,objectives,training_state}.py outputs/gan_stage_joint4_4gpu/joint4_20261019T122237_4gpu_mb16/source/*.py
