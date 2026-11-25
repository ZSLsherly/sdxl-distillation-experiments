set -euo pipefail
hostname
nvidia-smi --query-gpu=index,name,memory.free,utilization.gpu --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv
