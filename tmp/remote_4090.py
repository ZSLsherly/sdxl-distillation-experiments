import sys, re
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / '.image_runtime_pkgs'))
import paramiko
raw = Path(r'C:\Users\sherl\Desktop\login.txt').read_text(encoding='utf-8-sig')
u,p = next((u,p.strip()) for ip,u,p in re.findall(r'IP:\s*(\d+\.\d+\.\d+\.\d+)\s+User:\s*(\S+)\s+Password:\s*([^\r\n]+)',raw) if ip=='172.18.132.41')
c=paramiko.SSHClient(); c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('172.18.132.41',username=u,password=p,timeout=15,look_for_keys=False,allow_agent=False)
c.get_transport().set_keepalive(20)
_,out,_=c.exec_command(Path(sys.argv[1]).read_text(encoding='utf-8'),timeout=600)
out.channel.set_combine_stderr(True)
for line in out: print(line.rstrip(),flush=True)
rc=out.channel.recv_exit_status();c.close();sys.exit(rc)
