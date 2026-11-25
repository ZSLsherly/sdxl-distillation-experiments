set -eu
. /root/sdxl_research/env.sh
ps -p 61613,42356 -o pid,ppid,etime,args
tail -n 5 "$SDXL_WORK/logs/uniform_after_bc.log"
python - <<'PY'
from pathlib import Path
import json,hashlib
r=Path('/root/sdxl_research');o=r/'outputs/turning_BC_paired_quota_20261102'
assert not (o/'uniform_control').exists(), 'Control unexpectedly started before C finished'
c=json.loads((o/'C/config.json').read_text());print('C_STATUS',(o/'C/status.json').read_text())
for n,h in json.loads((o/'C/source_sha256.json').read_text()).items():
 p=Path(c['turning_profile']) if n=='turning_distribution.json' else r/'code/sdxl_research'/n
 assert hashlib.sha256(p.read_bytes()).hexdigest()==h,n
print('RUNNING_TRAINING_SOURCE_UNCHANGED')
print('UNIFORM_CONFIG', (r/'code/sdxl_research/configs/uniform_control_a800_g160.json').read_text())
PY
