set -e
source /data5/shenglin.zeng/env.sh
cd /data5/shenglin.zeng/sdxl_crossdistill
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
free -h
df -h /data5
find datasets models -maxdepth 3 -type f -printf '%p %s\n' | head -n 55
python - <<'PY'
import inspect
from torch.distributed._composable.fsdp import CPUOffloadPolicy
print(inspect.signature(CPUOffloadPolicy))
import transformers
print('transformers',transformers.__version__)
PY
