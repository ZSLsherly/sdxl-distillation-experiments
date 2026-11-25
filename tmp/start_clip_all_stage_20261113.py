import sys,json
from pathlib import Path
from clip_all_remote_20261113 import connect,auth
c=connect('172.18.33.60')
p='/data5/shenglin.zeng/tools/eval_transfer_20261025/clip_all_stage_20261113.py'
with c.open_sftp() as s:s.put('tmp/clip_all_stage_20261113.py',p)
w,o,_=c.exec_command('/data5/shenglin.zeng/envs/python-runtime/bin/python3.12 -u '+p+' --detach'+(' --remaining-only' if '--remaining-only' in sys.argv else ''));o.channel.set_combine_stderr(True)
w.write(json.dumps({h:auth[h] for h in ['172.18.132.41','172.18.32.78','172.18.32.81']})+'\n');w.flush();w.channel.shutdown_write()
for line in o:print(line.strip(),flush=True)
sys.exit(o.channel.recv_exit_status())
