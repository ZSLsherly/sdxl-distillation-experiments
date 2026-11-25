set -eu
source /data5/shenglin.zeng/env.sh
python - <<'PY'
import json
from pathlib import Path
from huggingface_hub import hf_hub_download
root=Path('/data5/shenglin.zeng/sdxl_crossdistill')
for p in [root/'exports/coco_uniform_g160_20261025',Path('/data5/shenglin.zeng/tools')]:
 if p.exists():print('EXPORTS',str(p),[(x.name,x.stat().st_size) for x in p.iterdir()])
try:
 path=hf_hub_download('stabilityai/stable-diffusion-3-medium-diffusers','transformer/config.json',cache_dir=root/'cache/sd3_access_probe')
 print('SD3_CONFIG_ACCESS_OK',path)
except Exception as e:
 print('SD3_CONFIG_ACCESS_ERROR',type(e).__name__,str(e)[:500])
PY
