from clip_all_remote_20261113 import connect
from pathlib import Path
import json
c=connect('172.18.32.81');r='/data3/shenglin.zeng/clip_missing_20261113';py='/data3/shenglin.zeng/sdxl_turning_ablation_20261028/env/bin/python'
with c.open_sftp() as s:
 for name,local in [('run_queue.py','tmp/clip_all_queue_20261113.py'),('report.py','tmp/clip_all_report_20261113.py')]:s.put(local,r+'/code/'+name)
cmd=f'''cd {r}
{py} -m py_compile code/worker.py code/run_queue.py code/report.py
for g in 5; do
 if test -s queue_$g.pid && kill -0 "$(cat queue_$g.pid)" 2>/dev/null; then echo EXISTING_$g; else
 nohup env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 {py} -u code/run_queue.py "$g" >queue_$g.log 2>&1 </dev/null & echo $! >queue_$g.pid
 fi
done
sleep 3
cat queue_5.log
cat queue_5.pid
'''
_,o,_=c.exec_command(cmd);o.channel.set_combine_stderr(True)
for l in o:print(l.rstrip(),flush=True)
raise SystemExit(o.channel.recv_exit_status())
