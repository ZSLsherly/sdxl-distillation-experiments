import hashlib,json,os,re,shlex,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'.image_runtime_pkgs'))
import paramiko
local=Path('output/sdxl-crossdistill/dmd2_official_recipe_4gpu_20261123').resolve();host='172.18.132.41';remote='/data1/shenglin.zeng/dmd2_official_recipe_4gpu_20261123'
raw=Path('C:/Users/sherl/Desktop/login.txt').read_text(encoding='utf-8-sig')
auth={ip:(u,p.strip()) for ip,u,p in re.findall(r'IP:\s*(\d+\.\d+\.\d+\.\d+)\s+User:\s*(\S+)\s+Password:\s*([^\r\n]+)',raw)}
rc=str(Path('tmp/rclone_runtime/rclone-v1.76.0-windows-amd64/rclone.exe').resolve())
env=dict(os.environ,RCLONE_CONFIG='NUL',RCLONE_CONFIG_LAB_TYPE='sftp',RCLONE_CONFIG_LAB_HOST=host,RCLONE_CONFIG_LAB_USER=auth[host][0],RCLONE_CONFIG_LAB_PORT='22')
env['RCLONE_CONFIG_LAB_PASS']=subprocess.check_output([rc,'obscure','-'],input=auth[host][1]+'\n',text=True).strip()

dest=local/'短测证据';dest.mkdir(exist_ok=True)
subprocess.run([rc,'copy','lab:'+remote+'/runs/smoke_micro8',str(dest),'--exclude','*.pt','--log-level','ERROR','--retries','2'],env=env,check=True)
c=paramiko.SSHClient();c.set_missing_host_key_policy(paramiko.AutoAddPolicy());c.connect(host,username=auth[host][0],password=auth[host][1],timeout=15,look_for_keys=False,allow_agent=False)
checks=[]
for p in dest.rglob('*'):
 if not p.is_file():continue
 source=remote+'/runs/smoke_micro8/'+p.relative_to(dest).as_posix();_,o,_=c.exec_command('sha256sum '+shlex.quote(source));got=o.read().decode().split()[0]
 sha=hashlib.sha256(p.read_bytes()).hexdigest();assert o.channel.recv_exit_status()==0 and sha==got
 checks.append(dict(source_host=host,source=source,destination_host='local Windows workstation',destination=str(p),sha256=sha,verified=True))
(local/'短测取回校验.txt').write_text('rclone取回短测日志与图片，不含权重；逐文件SHA256一致。\n'+json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf-8')
print('RCLONE_FETCH_SHA256_VERIFIED',len(checks));c.close()

