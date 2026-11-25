import sys,re
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'.image_runtime_pkgs'))
import paramiko
raw=Path(r'C:\Users\sherl\Desktop\login.txt').read_text(encoding='utf-8-sig')
creds={ip:(u,p.strip()) for ip,u,p in re.findall(r'IP:\s*(\d+\.\d+\.\d+\.\d+)\s+User:\s*(\S+)\s+Password:\s*([^\r\n]+)',raw)}
ip,mode=sys.argv[1:3]
c=paramiko.SSHClient();c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
u,p=creds[ip];c.connect(ip,username=u,password=p,timeout=15,look_for_keys=False,allow_agent=False)
try:
 if mode=='exec':
  cmd=Path(sys.argv[3]).read_text(encoding='utf-8-sig')
  _,o,e=c.exec_command(cmd,timeout=1800);o.channel.set_combine_stderr(True)
  for line in o: print(line.rstrip(),flush=True)
  sys.exit(o.channel.recv_exit_status())
 elif mode=='put':
  with c.open_sftp() as s: s.put(sys.argv[3],sys.argv[4])
 elif mode=='get':
  with c.open_sftp() as s: s.get(sys.argv[3],sys.argv[4])
finally: c.close()
