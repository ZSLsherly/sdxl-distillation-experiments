set -e
source /data5/shenglin.zeng/env.sh
python - <<'PY'
import inspect
from torch.distributed.tensor import DTensor
s=inspect.getsource(DTensor.from_local)
print(s[-5000:])
print(inspect.signature(DTensor))
PY
