set -eu
R=/data1/shenglin.zeng/dmd2_official_recipe_4gpu_20261123
cd "$R/code"
"$R/env/bin/python" - <<'PY'
from pathlib import Path
import json,hashlib
from run_guard import check_idle
r=Path('..');c=json.loads((r/'config.json').read_text());assert c['smoke_micro_candidates']==[16]
check_idle(c['gpu_uuids']);assert not (r/'runs/formal_g160').exists()
assert not (r/'runs/smoke_micro16').exists()
old=json.loads((r/'source_manifest.json').read_text())
for name,sha in old.items():
 if name!='config.json':assert hashlib.sha256((r/name).read_bytes()).hexdigest()==sha,name
old['config.json']=hashlib.sha256((r/'config.json').read_bytes()).hexdigest()
(r/'source_manifest.json').write_text(json.dumps(old,indent=2))
PY
nohup "$R/env/bin/python" -u "$R/code/run_guard.py" --root "$R" --run formal_g160 >> "$R/logs/queue.log" 2>&1 < /dev/null &
echo "MICRO16_QUEUE_PID=$!"
sleep 2
tail -n 3 "$R/logs/queue.log"
