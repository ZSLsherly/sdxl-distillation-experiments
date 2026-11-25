import sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'.image_runtime_pkgs'))
import paramiko
client=paramiko.SSHClient();client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('connect.nma1.seetacloud.com',port=14892,username='root',password=sys.stdin.readline().strip(),timeout=20,look_for_keys=False,allow_agent=False)
mode,src=sys.argv[1:3]
if mode=='exec':
 _,out,_=client.exec_command(Path(src).read_text(encoding='utf-8-sig'),timeout=600)
 out.channel.set_combine_stderr(True)
 for line in out:print(line.rstrip(),flush=True)
 rc=out.channel.recv_exit_status();print('EXIT',rc);client.close();sys.exit(rc)
with client.open_sftp() as ftp:
 if mode=='put':ftp.put(src,sys.argv[3])
 else:ftp.get(src,sys.argv[3])
print(mode,'complete');client.close()
