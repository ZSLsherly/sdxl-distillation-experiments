set -euo pipefail
source /data5/shenglin.zeng/env.sh
cd /data5/shenglin.zeng/sdxl_crossdistill
date -Is
nvidia-smi -i 0,2 --query-gpu=index,uuid,pci.bus_id,name,memory.total,memory.free,utilization.gpu --format=csv
nvidia-smi -i 0,2 --query-compute-apps=gpu_uuid,pid,used_memory --format=csv
df -h /data5/shenglin.zeng
free -h
python - <<'PY'
import hashlib,json
from pathlib import Path
p=Path('validation/modules_prepared_20261022/sdxl_research')
m=json.loads((p/'source_manifest.json').read_text())
assert all(hashlib.sha256((p/k).read_bytes()).hexdigest()==v for k,v in m.items())
print('VALIDATED_SOURCE_FILES',len(m))
for asset in ['models/sdxl-base-1.0/unet','models/dinov2-small','datasets/parallel_10k_seed20261018/prepared.pt']:
 print('ASSET_EXISTS',asset,Path(asset).exists())
print('SOURCE_MANIFEST_SHA256',hashlib.sha256((p/'source_manifest.json').read_bytes()).hexdigest())
PY
