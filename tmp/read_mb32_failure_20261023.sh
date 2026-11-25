source /data5/shenglin.zeng/env.sh
cd /data5/shenglin.zeng/sdxl_crossdistill
python - <<'PY'
from pathlib import Path
import json
text=Path('logs/microbatch32_20261023_probe.log').read_text()
lines=text.splitlines()
print('\n'.join(lines[:180]))
p=Path('outputs/microbatch32_20261023/resource_monitor.jsonl')
rows=[json.loads(x) for x in p.read_text().splitlines()]
print('ANY_FOREIGN',any(x['foreign'] for x in rows))
PY
