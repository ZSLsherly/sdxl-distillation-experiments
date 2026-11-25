cd /data5/shenglin.zeng/a50k_20261124
python3 - <<'PY'
import json
from pathlib import Path
for p in Path('outputs/A50K_singlepass/eval').glob('*_g000616_periodic/rank*.json'):
 for r in json.loads(p.read_text())['samples']:
  if r['split']=='holdout': print(r['original_index'],repr(r['prompt'][:500]))
PY
