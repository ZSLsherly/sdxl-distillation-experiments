/data1/shenglin.zeng/dmd2_matched_10k_20261119/env/bin/python - <<'PY'
import json,importlib.metadata,platform
from pathlib import Path
r=Path('/data1/shenglin.zeng/dmd2_matched_10k_20261119/coco10k_official_20261121')
v={k:importlib.metadata.version(k) for k in ['torch','torchvision','diffusers','transformers','clean-fid','Pillow','numpy','scipy']};v['python']=platform.python_version();(r/'environment_versions.json').write_text(json.dumps(v,indent=2));print(json.dumps(v))
PY
