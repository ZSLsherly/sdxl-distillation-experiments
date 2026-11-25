set -eu
source /data5/shenglin.zeng/env.sh
export CUDA_VISIBLE_DEVICES=''
python - <<'PY'
import importlib.util,os
from pathlib import Path
print('PACKAGES',{n:bool(importlib.util.find_spec(n)) for n in ['lpips','torchvision','transformers','safetensors']})
root=Path('/data5/shenglin.zeng/sdxl_crossdistill')
for p in [root/'models/dinov2-small',root/'models/metrics',Path('/data5/shenglin.zeng/cache/torch/hub/checkpoints')]:
 if p.exists():
  for f in p.glob('*'):print('FILE',f,f.stat().st_size)
print('TORCH_CACHE',os.environ.get('TORCH_HOME','unset'))
PY
