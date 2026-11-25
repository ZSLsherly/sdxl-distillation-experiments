import sys,re,json
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'.image_runtime_pkgs'))
import paramiko
raw=Path(r'C:/Users/sherl/Desktop/login.txt').read_text(encoding='utf-8-sig')
creds={ip:(u,p.strip()) for ip,u,p in re.findall(r'IP:\s*(\d+\.\d+\.\d+\.\d+)\s+User:\s*(\S+)\s+Password:\s*([^\r\n]+)',raw)}
c=paramiko.SSHClient();c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
u,p=creds['172.18.33.60'];c.connect('172.18.33.60',username=u,password=p,timeout=15,look_for_keys=False,allow_agent=False)
c.get_transport().set_keepalive(20)
stdin,out,_=c.exec_command('/data5/shenglin.zeng/envs/crossdistill/bin/python -u /data5/shenglin.zeng/sdxl_crossdistill/exports/turning_20261026/transfer.py',timeout=3600)
du,dp=creds['172.18.32.78'];stdin.write(json.dumps({'user':du,'password':dp})+'\n');stdin.flush();stdin.channel.shutdown_write()
out.channel.set_combine_stderr(True)
for line in out:print(line.rstrip(),flush=True)
rc=out.channel.recv_exit_status();print('TRANSFER_EXIT',rc);c.close();raise SystemExit(rc)
