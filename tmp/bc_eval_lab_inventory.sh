set -eu
p=/data1/shenglin.zeng/sdxl_eval_uniform_20261025
date -Is
nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader,nounits
for f in "$p/code/evaluate.py" "$p/env/bin/python" "$p/outputs/coco10k/manifest.json" "$p/outputs/coco10k/metrics" "$p/outputs/coco10k/features/real" "$p/reference";do ls -ld "$f" 2>/dev/null || true;done
df -h "$p"
ls "$p"; ls "$p/outputs/coco10k"; ls "$p/reference"
