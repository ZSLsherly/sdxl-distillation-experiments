import json,re,sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'.image_runtime_pkgs'))
import paramiko
raw=Path('C:/Users/sherl/Desktop/login.txt').read_text(encoding='utf-8-sig')
creds={ip:(u,p.strip()) for ip,u,p in re.findall(r'IP:\s*(\d+\.\d+\.\d+\.\d+)\s+User:\s*(\S+)\s+Password:\s*([^\r\n]+)',raw)}
host='172.18.33.60';u,p=creds[host]
c=paramiko.SSHClient();c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(host,username=u,password=p,timeout=15,look_for_keys=False,allow_agent=False)
mode=sys.argv[1]
if mode=='exec':
 _,o,_=c.exec_command(Path(sys.argv[2]).read_text(encoding='utf-8-sig'),timeout=240)
 o.channel.set_combine_stderr(True)
 sys.stdout.reconfigure(encoding="utf-8", errors="replace");print(o.read().decode(errors="replace"),flush=True)
 status=o.channel.recv_exit_status()
else:
 with c.open_sftp() as f:
  (f.put if mode=='put' else f.get)(sys.argv[2],sys.argv[3])
 status=0
c.close();sys.exit(status)


