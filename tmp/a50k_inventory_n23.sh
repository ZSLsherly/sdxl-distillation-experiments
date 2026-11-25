date
find /data1/shenglin.zeng -maxdepth 5 -type f \( -name prepared.pt -o -name 'vae_latents_*.pt' -o -name validation.json \) -printf '%p %s\n' 2>/dev/null | head -65
find /data1/shenglin.zeng -maxdepth 4 -type d -name sdxl_research 2>/dev/null | head -12
df -h /data1
nvidia-smi --query-gpu=index,uuid,memory.free,utilization.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader
