nvidia-smi -i 1,6 --query-gpu=index,uuid,memory.used,memory.total,utilization.gpu --format=csv,noheader
nvidia-smi -i 1,6 --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader
df -h /data1
find /data1/shenglin.zeng -maxdepth 5 -type f \( -name results.json -o -name manifest.json -o -name model_index.json \) | head -80
/data1/shenglin.zeng/sdxl_eval_uniform_20261025/env/bin/python -c 'import torch,transformers,diffusers; print(torch.__version__,transformers.__version__,diffusers.__version__)'
