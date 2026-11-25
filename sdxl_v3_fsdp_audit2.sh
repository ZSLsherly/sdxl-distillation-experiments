set -e
source /data5/shenglin.zeng/env.sh
cd /data5/shenglin.zeng/sdxl_crossdistill
python - <<'PY'
import inspect
from torch.distributed._composable.fsdp._fsdp_collectives import foreach_reduce
s=inspect.getsource(foreach_reduce)
start=s.find('offload_to_cpu')
print(s[max(0,start-800):start+3100])
PY
tail -n 12 logs/gan_v3_preflight.log
