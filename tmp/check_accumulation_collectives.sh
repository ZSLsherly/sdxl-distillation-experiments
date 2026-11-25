source /data5/shenglin.zeng/env.sh
export CUDA_VISIBLE_DEVICES=''
python - <<'PY'
import inspect
from torch.distributed._composable.fsdp import _fsdp_collectives
lines=inspect.getsource(_fsdp_collectives).splitlines()
shown=set()
for i,l in enumerate(lines):
 if 'offload_to_cpu' in l or 'grad += ' in l or 'grad_data.add_' in l:
  for j in range(max(0,i-4),min(len(lines),i+28)):
   if j not in shown:print(f'{j+1}: {lines[j]}');shown.add(j)
PY
