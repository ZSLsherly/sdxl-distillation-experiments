import hashlib,json,os,re,shlex,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'.image_runtime_pkgs'))
import paramiko
local=Path('output/sdxl-crossdistill/dmd2_matched_20261119').resolve();host='172.18.132.41';remote='/data1/shenglin.zeng/dmd2_matched_10k_20261119'
raw=Path('C:/Users/sherl/Desktop/login.txt').read_text(encoding='utf-8-sig')
auth={ip:(u,p.strip()) for ip,u,p in re.findall(r'IP:\s*(\d+\.\d+\.\d+\.\d+)\s+User:\s*(\S+)\s+Password:\s*([^\r\n]+)',raw)}
rc=str(Path('tmp/rclone_runtime/rclone-v1.76.0-windows-amd64/rclone.exe').resolve())
env=dict(os.environ,RCLONE_CONFIG='NUL',RCLONE_CONFIG_LAB_TYPE='sftp',RCLONE_CONFIG_LAB_HOST=host,RCLONE_CONFIG_LAB_USER=auth[host][0],RCLONE_CONFIG_LAB_PORT='22')
env['RCLONE_CONFIG_LAB_PASS']=subprocess.check_output([rc,'obscure','-'],input=auth[host][1]+'\n',text=True).strip()

names=['config.json','code/run_guard.py','code/gpu_probe.py','source_manifest.json']
c=paramiko.SSHClient();c.set_missing_host_key_policy(paramiko.AutoAddPolicy());c.connect(host,username=auth[host][0],password=auth[host][1],timeout=15,look_for_keys=False,allow_agent=False)
checks=[]
for name in names:
 p=local/name;dest=remote+'/'+name
 subprocess.run([rc,'copyto',str(p),'lab:'+dest,'--log-level','ERROR'],env=env,check=True)
 _,o,_=c.exec_command('sha256sum '+shlex.quote(dest));got=o.read().decode().split()[0]
 sha=hashlib.sha256(p.read_bytes()).hexdigest();assert o.channel.recv_exit_status()==0 and sha==got
 checks.append(dict(source_host='local Windows workstation',source=str(p),destination_host=host,destination=dest,sha256=sha,verified=True))
(local/'重启代码传输校验.txt').write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf-8');c.close();print('FINAL_RECORDS_SHA256_VERIFIED',len(checks))
