"""Source host transfer. Destination credentials arrive via SSH stdin only."""
import sys,json
from pathlib import Path
sys.path.insert(0,'/data5/shenglin.zeng/tools/eval_transfer_20261025')
import paramiko
req=json.loads(sys.stdin.readline())
source=Path('/data5/shenglin.zeng/sdxl_crossdistill/exports/turning_20261026')
dest='/data1/shenglin.zeng/teacher_turning_20261026'
c=paramiko.SSHClient();c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('172.18.32.78',username=req['user'],password=req.pop('password'),timeout=15,look_for_keys=False,allow_agent=False)
c.get_transport().set_keepalive(20)
with c.open_sftp() as s:
    for path in ['/data1/shenglin.zeng',dest,dest+'/transfers',dest+'/tmp',dest+'/cache']:
        try:s.mkdir(path)
        except OSError:s.stat(path)
    for name in ('environment.tar.gz','sdxl_sampling.tar','manifest.json'):
        p=source/name
        last=[-1]
        def progress(done,total):
            n=done//(1<<30)
            if n!=last[0]:print('TRANSFER',name,round(done/2**30,2),'/',round(total/2**30,2),'GiB',flush=True);last[0]=n
        s.put(str(p),dest+'/transfers/'+name+'.partial',callback=progress)
        assert s.stat(dest+'/transfers/'+name+'.partial').st_size==p.stat().st_size
        s.posix_rename(dest+'/transfers/'+name+'.partial',dest+'/transfers/'+name)
        print('TRANSFERRED',name,p.stat().st_size,flush=True)
c.close()
