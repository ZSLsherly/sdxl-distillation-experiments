import sys,re
from pathlib import Path
sys.path.insert(0,'.runtime_pkgs')
import paramiko
raw=Path(r'C:\Users\sherl\Desktop\login.txt').read_text(encoding='utf-8-sig')
creds={ip:(u,p.strip()) for ip,u,p in re.findall(r'IP:\s*(\d+\.\d+\.\d+\.\d+)\s+User:\s*(\S+)\s+Password:\s*([^\r\n]+)',raw)}
c=paramiko.SSHClient();c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
u,p=creds['172.18.33.60'];c.connect('172.18.33.60',username=u,password=p,timeout=15,look_for_keys=False,allow_agent=False)
cmd=Path(sys.argv[1]).read_text(encoding='utf-8')
_,o,e=c.exec_command(cmd,timeout=600)
o.channel.set_combine_stderr(True)
for line in o:
 print(line.rstrip(),flush=True)
print('EXIT',o.channel.recv_exit_status(),flush=True);c.close()
