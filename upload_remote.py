import sys,re
from pathlib import Path
sys.path.insert(0,'.runtime_pkgs')
import paramiko
raw=Path(r'C:\Users\sherl\Desktop\login.txt').read_text(encoding='utf-8-sig')
creds={ip:(u,p.strip()) for ip,u,p in re.findall(r'IP:\s*(\d+\.\d+\.\d+\.\d+)\s+User:\s*(\S+)\s+Password:\s*([^\r\n]+)',raw)}
c=paramiko.SSHClient();c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
u,p=creds['172.18.33.60'];c.connect('172.18.33.60',username=u,password=p,timeout=15,look_for_keys=False,allow_agent=False)
with c.open_sftp() as s: s.put(sys.argv[1],sys.argv[2])
c.close();print('uploaded',sys.argv[2])
