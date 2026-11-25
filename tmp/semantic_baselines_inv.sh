set -eu
for d in /data*/shenglin.zeng; do if test -d "$d"; then find "$d" \( -name unet.safetensors -o -path '*/images/sdxl' -o -path '*/images/dmd2' \) -print;fi;done
nvidia-smi --query-gpu=index,uuid,memory.free,utilization.gpu --format=csv,noheader,nounits
nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits
