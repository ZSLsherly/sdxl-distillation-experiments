R=/data/shenglin.zeng/a50k_20261123
du -sh "$R"/checkpoints/singlepass/*
for d in "$R"/checkpoints/singlepass/*; do ls "$d"/complete.json;done
find /data3/shenglin.zeng /data1/shenglin.zeng "$R" -maxdepth 7 -type f \( -name 'real_stats*.npz' -o -name '*inception*.pt' -o -name 'real*.npz' -o -name 'selection.json' -o -name 'evaluation_core.py' \) 2>/dev/null |head -50
nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader
