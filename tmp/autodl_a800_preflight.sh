set -eu
date -Is
nvidia-smi --query-gpu=index,name,memory.total,memory.free,utilization.gpu,uuid --format=csv,noheader,nounits
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_gpu_memory --format=csv,noheader,nounits
df -h / /root/autodl-tmp
cat /sys/fs/cgroup/memory.max
cat /sys/fs/cgroup/cpu.max
ls -l /root/sdxl_research/env.sh
. /root/sdxl_research/env.sh
python - <<'PY'
import torch
print('TORCH',torch.__version__,'CUDA',torch.version.cuda,'GPU_COUNT',torch.cuda.device_count())
PY
