"""Task-scoped SSH helper; credentials stay in memory and are never logged."""
import sys,re,json
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'.image_runtime_pkgs'))
import paramiko
raw=Path('C:/Users/sherl/Desktop/login.txt').read_text(encoding='utf-8-sig')
auth={ip:(u,p.strip()) for ip,u,p in re.findall(r'IP:\s*(\d+\.\d+\.\d+\.\d+)\s+User:\s*(\S+)\s+Password:\s*([^\r\n]+)',raw)}
def connect(host):
    c=paramiko.SSHClient();c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    u,p=auth[host];c.connect(host,username=u,password=p,timeout=15,look_for_keys=False,allow_agent=False)
    return c
if __name__=='__main__':
    c=connect(sys.argv[1]);mode=sys.argv[2]
    if mode=='exec':
        _,o,_=c.exec_command(Path(sys.argv[3]).read_text(encoding='utf-8'),timeout=600)
        o.channel.set_combine_stderr(True)
        for line in o: print(line.rstrip(),flush=True)
        rc=o.channel.recv_exit_status();c.close();sys.exit(rc)
    with c.open_sftp() as s:
        if mode=='get':s.get(sys.argv[3],sys.argv[4])
        elif mode=='put':s.put(sys.argv[3],sys.argv[4])
    c.close()
