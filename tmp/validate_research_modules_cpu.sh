set -euo pipefail
source /data5/shenglin.zeng/env.sh
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
dest=/data5/shenglin.zeng/sdxl_crossdistill/validation/modules_prepared_20261022
mkdir -p "$dest"
tar -xzf /data5/shenglin.zeng/sdxl_crossdistill/sdxl_modules_prepared_20261022.tar.gz -C "$dest"
cd "$dest/sdxl_research"
nice -n 19 python -m unittest discover -v 2>&1 | tee "$dest/cpu_tests.log"
nice -n 19 python - <<'PY'
import hashlib,json
from pathlib import Path
manifest=json.loads(Path('source_manifest.json').read_text())
assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in manifest.items())
import torch
print('SOURCE_HASHES_MATCH',len(manifest),'visible_GPUs',torch.cuda.device_count())
PY
