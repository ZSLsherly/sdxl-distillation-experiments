set -e
R=/data1/shenglin.zeng/dmd2_matched_10k_20261119
cd "$R"
"$R/env/bin/python" - <<'PY'
import json,hashlib
from pathlib import Path
r=Path.cwd();m=json.loads((r/'source_manifest.json').read_text())
for name,expected in m.items():assert hashlib.sha256((r/name).read_bytes()).hexdigest()==expected,name
assert not (r/'STOP_REQUEST').exists()
print('SOURCE_SHA256_PASS',len(m))
PY
nohup "$R/env/bin/python" -u "$R/code/run_guard.py" --root "$R" --run formal_g160_restart_20261120 --reuse-smoke > "$R/logs/queue_restart_20261120.log" 2>&1 < /dev/null &
echo "GUARD_PID=$!"
sleep 3
cat "$R/logs/queue_restart_20261120.log"
df -h /data1
