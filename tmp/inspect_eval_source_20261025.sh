set -euo pipefail
source /data5/shenglin.zeng/env.sh
export CUDA_VISIBLE_DEVICES=''
cd /data5/shenglin.zeng/sdxl_crossdistill
python - <<'PY'
import sys,importlib.util,json
from pathlib import Path
print('ENV',sys.prefix,'conda_pack',bool(importlib.util.find_spec('conda_pack')))
for p in ['models/sdxl-base-1.0','models/metrics/clean_fid','outputs/coco10k_g160_20261020/features/real']:
 root=Path(p)
 print('ASSET',p, sum(f.stat().st_size for f in root.rglob('*') if f.is_file())/2**30)
print('INCEPTION_FILES',[str(p) for p in Path('models/metrics/clean_fid').rglob('*') if p.is_file()])
PY
du -sh /data5/shenglin.zeng/envs/crossdistill
