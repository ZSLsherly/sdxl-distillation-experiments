R=/data3/shenglin.zeng/new_models_eval_20261114
ls -ld "$R"/env "$R"/reference/features/real "$R"/models/* "$R"/results/*/manifest.json
ls "$R"/reference/features/real |head -2
cat "$R"/node.json | head -c 2300
cd /data/shenglin.zeng/a50k_20261123/checkpoints/singlepass
python3 - <<'PY'
from pathlib import Path
import json
p=next(Path('.').glob('*/complete.json'));d=json.loads(p.read_text());print(d['config']['schedule'],d['progress'])
PY
