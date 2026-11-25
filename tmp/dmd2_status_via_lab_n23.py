"""Read DMD2 status over an authorized lab SSH jump; no credentials stored or printed."""
import sys,re
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'.image_runtime_pkgs'))
import paramiko
raw=Path('C:/Users/sherl/Desktop/login.txt').read_text(encoding='utf-8-sig')
auth={h:(u,p.strip()) for h,u,p in re.findall(r'IP:\s*(\d+\.\d+\.\d+\.\d+)\s+User:\s*(\S+)\s+Password:\s*([^\r\n]+)',raw)}
outer=paramiko.SSHClient();outer.set_missing_host_key_policy(paramiko.AutoAddPolicy())
inner=paramiko.SSHClient();inner.set_missing_host_key_policy(paramiko.AutoAddPolicy())
try:
 h='172.18.33.60';outer.connect(h,username=auth[h][0],password=auth[h][1],timeout=12,banner_timeout=20,look_for_keys=False,allow_agent=False)
 sock=outer.get_transport().open_channel('direct-tcpip',('172.18.132.41',22),('127.0.0.1',0),timeout=12)
 h='172.18.132.41';inner.connect(h,username=auth[h][0],password=auth[h][1],sock=sock,timeout=20,banner_timeout=30,look_for_keys=False,allow_agent=False)
 _,out,_=inner.exec_command(Path('tmp/dmd2_recipe_status_n23.sh').read_text(encoding='utf-8'),timeout=120)
 out.channel.set_combine_stderr(True)
 for line in out:print(line.rstrip(),flush=True)
 print('REMOTE_EXIT',out.channel.recv_exit_status())
finally:
 inner.close();outer.close()
