set -eu
/data1/shenglin.zeng/dmd2_official_recipe_20261121/env/bin/python - <<'PY'
import json
from pathlib import Path
r=Path('/data1/shenglin.zeng/dmd2_official_recipe_20261121')
p=r/'runs/smoke_micro4/status.json'
assert 'OOM_REDUCE_MICRO 4' in (r/'logs/queue.log').read_text()
s=json.loads(p.read_text());s.update(status='oom_microbatch_rejected',reason='GPU out of memory in next fake backward after G1/F1; retried micro2 at same global batch64')
p.write_text(json.dumps(s,indent=2))
print('RECORDED_MICRO4_OOM_NOT_CURRENT_RUN')
PY
