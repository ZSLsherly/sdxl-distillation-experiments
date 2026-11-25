set -eu
source /data5/shenglin.zeng/env.sh
tail -n 8 "$CROSSDISTILL_ROOT/image_crossdistill/logs/download.log"
python - <<'PY'
import inspect
from diffusers import SanaPipeline
s=inspect.getsource(SanaPipeline.encode_prompt)
print(s[-6000:])
s=inspect.getsource(SanaPipeline.__call__)
print(s[-8500:])
PY
