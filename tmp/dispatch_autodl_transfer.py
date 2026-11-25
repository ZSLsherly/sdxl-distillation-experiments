import sys,re,json
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'.image_runtime_pkgs'))
import paramiko
raw=Path('C:/Users/sherl/Desktop/login.txt').read_text(encoding='utf-8-sig')
creds={ip:(u,p.strip()) for ip,u,p in re.findall(r'IP:\s*(\d+\.\d+\.\d+\.\d+)\s+User:\s*(\S+)\s+Password:\s*([^\r\n]+)',raw)}
u,p=creds['172.18.33.60'];c=paramiko.SSHClient();c.set_missing_host_key_policy(paramiko.AutoAddPolicy());c.connect('172.18.33.60',username=u,password=p,timeout=15,look_for_keys=False,allow_agent=False)
script='''import sys,subprocess,json
request=sys.stdin.readline()
log=open('/data5/shenglin.zeng/sdxl_crossdistill/tmp/transfer_autodl_assets.log','a')
p=subprocess.Popen(['/data5/shenglin.zeng/envs/crossdistill/bin/python','-u','/data5/shenglin.zeng/sdxl_crossdistill/tmp/transfer_autodl_assets.py'],stdin=subprocess.PIPE,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
p.stdin.write(request.encode());p.stdin.close();print('TRANSFER_PID',p.pid)
'''
import shlex
stdin,out,err=c.exec_command('/data5/shenglin.zeng/envs/crossdistill/bin/python -c '+shlex.quote(script));stdin.write(json.dumps({'user':'root','password':sys.stdin.readline().strip()})+'\n');stdin.flush();stdin.channel.shutdown_write();print(out.read().decode());print(err.read().decode());c.close()
