set -e
source /data5/shenglin.zeng/env.sh
cd /data5/shenglin.zeng/sdxl_crossdistill
python - <<'PY'
import inspect
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed._composable.fsdp._fsdp_param_group import FSDPParamGroup
print(inspect.getsource(FSDPParamGroup.post_backward)[-9000:])
PY
