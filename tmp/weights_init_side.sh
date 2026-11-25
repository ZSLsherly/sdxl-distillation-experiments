set -eu
. /root/sdxl_research/env.sh
python - <<'PY'
import json,os
from pathlib import Path
p=Path(os.environ['SDXL_WORK'])/'outputs/c_mix1_dm234_20261103/model/config.json'
if p.exists():
 m=json.loads(p.read_text());print({k:m.get(k) for k in ('initialization','init_student','resume','run','variant')})
for p in Path('/proc').glob('[0-9]*/cmdline'):
 try:
  cmd=p.read_bytes().replace(b'\x00',b' ').decode(errors='replace')
  if ('torch.distributed.run' in cmd or ('/train.py' in cmd and 'python' in cmd)) and 'awk ' not in cmd:print(p.parent.name,cmd[:200])
 except OSError:pass
PY
