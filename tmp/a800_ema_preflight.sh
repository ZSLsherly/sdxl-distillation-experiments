set -eu
. /root/sdxl_research/env.sh
CUDA_VISIBLE_DEVICES='' python - <<'PY'
from torch.distributed._composable.fsdp import FSDPModule
import inspect
print(inspect.getsource(FSDPModule.reshard))
PY
cat "$SDXL_ROOT/outputs/turning_BC_paired_quota_20261102/uniform_control/status.json"
