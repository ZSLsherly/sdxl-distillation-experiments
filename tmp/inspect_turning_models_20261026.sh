set -eu
source /data5/shenglin.zeng/env.sh
python - <<'PY'
from pathlib import Path
import json
root=Path('/data5/shenglin.zeng')
for path in [root/'models',root/'sdxl_crossdistill/models',root/'cache/huggingface/hub',root/'sdxl_crossdistill/cache/hub']:
 if path.exists():
  print('MODEL_DIRECTORY',str(path),[p.name for p in path.iterdir() if p.is_dir()])
out=root/'sdxl_crossdistill/outputs/boundary_dmd_10k_a100_20261025_v1'
for p in [out/'workflow_error.json',out/'workflow_complete.json',out/'train/status.json',out/'short_retry1/benchmark.json']:
 if p.exists():print(str(p),p.read_text()[:3500])
PY
