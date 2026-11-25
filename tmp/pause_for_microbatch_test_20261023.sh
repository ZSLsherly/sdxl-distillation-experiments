set -euo pipefail
source /data5/shenglin.zeng/env.sh
cd /data5/shenglin.zeng/sdxl_crossdistill
python - <<'PY'
from pathlib import Path
p=Path('outputs/joint4_uniform_10k_2gpu_20261023/train')
assert p.joinpath('config.json').exists()
p.joinpath('STOP_REQUEST').write_text('User requested testing micro-batch32; save at completed cycle then release GPUs.\n')
print('REQUESTED_GRACEFUL_STOP',p)
PY
