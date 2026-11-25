"""Operate only explicit research files on a user-authorized SSH host."""
import sys,re
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'.image_runtime_pkgs'))
import paramiko
raw=Path(r'C:/Users/sherl/Desktop/login.txt').read_text(encoding='utf-8-sig')
creds={ip:(u,p.strip()) for ip,u,p in re.findall(r'IP:\s*(\d+\.\d+\.\d+\.\d+)\s+User:\s*(\S+)\s+Password:\s*([^\r\n]+)',raw)}
ip,mode,src=sys.argv[1:4];u,p=creds[ip]
c=paramiko.SSHClient();c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(ip,username=u,password=p,timeout=12,look_for_keys=False,allow_agent=False)
if mode=='exec':
 _,o,_=c.exec_command(Path(src).read_text(encoding='utf-8'),timeout=600)
 o.channel.set_combine_stderr(True)
 for line in o:print(line.rstrip(),flush=True)
 print('EXIT',o.channel.recv_exit_status())
else:
 with c.open_sftp() as s:
  if mode=='put':s.put(src,sys.argv[4])
  elif mode=='get':s.get(src,sys.argv[4])
  else:raise ValueError(mode)
 print(mode,'complete')
c.close()
