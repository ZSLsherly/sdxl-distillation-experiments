source /data5/shenglin.zeng/env.sh
export CUDA_VISIBLE_DEVICES=''
python - <<'PY'
import inspect
from torch.distributed._composable.fsdp import _fsdp_param_group,_fsdp_param
for module in (_fsdp_param_group,_fsdp_param):
 lines=inspect.getsource(module).splitlines()
 print('SOURCE',module.__file__)
 hits=[i for i,l in enumerate(lines) if 'accumulat' in l.lower() or 'grad_offload' in l.lower()]
 shown=set()
 for i in hits:
  for j in range(max(0,i-3),min(len(lines),i+7)):
   if j not in shown:print(f'{j+1}: {lines[j]}');shown.add(j)
PY
