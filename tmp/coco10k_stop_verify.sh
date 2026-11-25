set -eu
source /data5/shenglin.zeng/env.sh
python - <<'PY'
from pathlib import Path
import os,signal
parent=Path('/proc/3839823')
if parent.exists():
    ids=(parent/'task/3839823/children').read_text().split()
    for n in ids:
        cmd=(Path('/proc')/n/'cmdline').read_bytes()
        if b'sdxl_coco10k/validate.py' in cmd:
            os.kill(int(n),signal.SIGTERM)
            print('STOPPED_OWN_PREFLIGHT',n)
PY
