set -euo pipefail
source /data5/shenglin.zeng/env.sh
export CUDA_VISIBLE_DEVICES=''
cd /data5/shenglin.zeng/sdxl_crossdistill
python - <<'PY'
from pathlib import Path
import json,hashlib
r=Path.cwd();run=r/'outputs/joint4_uniform_10k_2gpu_mb16_20261023/train'
old=r/'outputs/joint4_uniform_10k_2gpu_20261023/train'
for name,p in [('reference',run/'config.json'),('initial_phase',old/'config.json')]:
 print(name,json.dumps(json.loads(p.read_text())))
source=run.parent/'code/sdxl_research'
hashes=json.loads((run/'source_sha256.json').read_text())
assert all(hashlib.sha256((source/n).read_bytes()).hexdigest()==h for n,h in hashes.items())
print('FROZEN_SOURCE_MATCHES_TRAINING',len(hashes))
for rel in ['datasets/parallel_10k_seed20261018','models/dinov2-small']:
 p=r/rel
 print('ASSET',rel,[(str(q.relative_to(p)),q.stat().st_size) for q in p.rglob('*') if q.is_file()])
print('SOURCE_PY_FILES',[p.name for p in source.glob('*.py')])
PY
