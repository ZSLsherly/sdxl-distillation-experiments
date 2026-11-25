set -eu
. /root/sdxl_research/env.sh
python - <<'PY'
from pathlib import Path
import os,json
root=Path('/root/sdxl_research'); pilot=root/'outputs/fake_tracking_pilot_20261102'; proc=Path('/proc/79570')
status=json.loads((root/'outputs/turning_BC_paired_quota_20261102/uniform_control/status.json').read_text()); print('uniform',status['status'],status['student_updates'])
assert status['status']=='running'
assert proc.exists() and proc.joinpath('cmdline').read_bytes().split(b'\0')[2]==b'run_fake_tracking_pilot.py'
assert proc.joinpath('cwd').resolve()==(root/'tmp/fake_tracking_pilot_20261102/code').resolve()
assert not proc.joinpath('task/79570/children').read_text().strip()
assert not any((pilot/name).exists() for name in ('control','mix_1pct'))
(pilot/'STOP_REQUEST').touch();os.kill(79570,15);print('Original waiting-only pilot disarmed; no GPU pilot started.')
PY
