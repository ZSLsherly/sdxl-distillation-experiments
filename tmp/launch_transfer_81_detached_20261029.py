"""Detach the authorized transfer, passing credentials only through a pipe."""
import json, subprocess, sys
from pathlib import Path
request=json.loads(sys.stdin.readline())
root=Path('/data5/shenglin.zeng/sdxl_crossdistill')
script=root/'tmp/transfer_81_assets_20261028.py'
logpath=root/'tmp/transfer_81_assets_20261029.log'
with logpath.open('x') as log:
    p=subprocess.Popen(['nice','-n','19','ionice','-c','2','-n','7',sys.executable,'-u',str(script)],
                       stdin=subprocess.PIPE,stdout=log,stderr=subprocess.STDOUT,
                       start_new_session=True,close_fds=True,text=True)
    p.stdin.write(json.dumps(request)+'\n');p.stdin.close()
(root/'tmp/transfer_81_assets_20261029.pid').write_text(str(p.pid))
print('TRANSFER_PID',p.pid,'LOG',str(logpath),flush=True)
