set -eu
ROOT=/data5/shenglin.zeng
source "$ROOT/env.sh"
printf 'GPU STATUS\n'
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv
printf '\nDISK\n'
df -h "$ROOT"
printf '\nIMAGE LIBRARIES\n'
python - <<'PY'
import torch,diffusers
print(torch.__version__,diffusers.__version__)
from diffusers import SanaPipeline, StableDiffusionPipeline
print('Sana and Stable Diffusion pipeline imports OK')
PY
