python3 - <<'PY'
import os,signal
from pathlib import Path
target='/data5/shenglin.zeng/tools/eval_transfer_20261025/clip_all_stage_20261113.py'
for p in Path('/proc').glob('[0-9]*/cmdline'):
 try:args=p.read_bytes().split(b'\0')
 except OSError:continue
 if target.encode() in args and 'python' in args[0].decode(errors='ignore'):
  os.kill(int(p.parent.name),signal.SIGTERM);print('STOPPED_OWN_TRANSFER',p.parent.name)
PY
