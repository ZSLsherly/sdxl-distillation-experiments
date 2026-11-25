set -eu
. /root/sdxl_research/env.sh
ps -p 61613,79570 -o pid,ppid,etime,args
tail -n 4 "$SDXL_WORK/logs/fake_tracking_pilot_queue.log"
python - <<'PY'
import json,hashlib
from pathlib import Path
r=Path('/root/sdxl_research');p=r/'outputs/turning_BC_paired_quota_20261102/uniform_control'
h=json.loads((p/'source_sha256.json').read_text());assert hashlib.sha256((r/'code/sdxl_research/train.py').read_bytes()).hexdigest()==h['train.py']
print('Current training source unchanged;', (p/'status.json').read_text())
PY
