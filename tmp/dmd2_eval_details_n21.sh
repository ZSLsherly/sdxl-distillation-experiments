R=/data1/shenglin.zeng/sdxl_eval_uniform_20261025
find "$R/reference" "$R/models/metrics" /data1/shenglin.zeng/clip_missing_20261113 -maxdepth 3 -type f | head -n 35
ls -l "$R/outputs/coco10k/features"
ls -lh "$R/outputs/semantic_joint4_uniform_1024/model"
ls -l /data1/shenglin.zeng/high30_parallel_eval_20261114
head -c 1800 /data1/shenglin.zeng/high30_parallel_eval_20261114/code/evaluation_core.py
"/data1/shenglin.zeng/dmd2_matched_10k_20261119/env/bin/python" - <<'PY'
import importlib.util,json
from pathlib import Path
for x in ['cleanfid','scipy','transformers','torchvision']:print(x,importlib.util.find_spec(x) is not None)
p=Path('/data1/shenglin.zeng/sdxl_eval_uniform_20261025/outputs/coco10k/manifest.json');m=json.loads(p.read_text());print('MANIFEST',m['n'],m['csv_sha256'],m['protocol']);print('CACHE_ROOTS', list(m['models']))
PY
