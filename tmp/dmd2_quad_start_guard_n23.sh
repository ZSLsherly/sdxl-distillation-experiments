set -eu
R=/data1/shenglin.zeng/dmd2_official_recipe_4gpu_20261123
cd "$R/code"
"$R/env/bin/python" - <<'PY'
import json,hashlib
from pathlib import Path
from run_guard import check_idle
r=Path('..');assert (r/'fsdp_precision_passed.json').is_file()
for name,sha in json.loads((r/'source_manifest.json').read_text()).items():assert hashlib.sha256((r/name).read_bytes()).hexdigest()==sha,name
check_idle(json.loads((r/'config.json').read_text())['gpu_uuids'])
assert not (r/'runs/formal_g160').exists()
PY
nohup "$R/env/bin/python" -u "$R/code/run_guard.py" --root "$R" --run formal_g160 >> "$R/logs/queue.log" 2>&1 < /dev/null &
echo "QUAD_QUEUE_PID=$!"
sleep 2
tail -n 4 "$R/logs/queue.log"
