import sys,re,json
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'.image_runtime_pkgs'))
import paramiko
raw=Path(r'C:\Users\sherl\Desktop\login.txt').read_text(encoding='utf-8-sig')
creds={ip:(u,p.strip()) for ip,u,p in re.findall(r'IP:\s*(\d+\.\d+\.\d+\.\d+)\s+User:\s*(\S+)\s+Password:\s*([^\r\n]+)',raw)}
c=paramiko.SSHClient();c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
u,p=creds['172.18.33.60'];c.connect('172.18.33.60',username=u,password=p,timeout=20,look_for_keys=False,allow_agent=False)
c.get_transport().set_keepalive(20)
stdin,stdout,stderr=c.exec_command('nice -n 19 ionice -c 2 -n 7 /data5/shenglin.zeng/envs/crossdistill/bin/python -u /data5/shenglin.zeng/sdxl_crossdistill/transfer_joint34_assets_20261025.py',timeout=600)
stdout.channel.set_combine_stderr(True)
u,p=creds['172.18.132.41'];stdin.write(json.dumps({'user':u,'password':p})+'\n');stdin.flush();stdin.channel.shutdown_write()
for line in stdout:print(line.rstrip(),flush=True)
rc=stdout.channel.recv_exit_status();c.close();sys.exit(rc)
