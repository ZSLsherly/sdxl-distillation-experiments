import sys,re,json
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'.image_runtime_pkgs'))
import paramiko
raw=Path(r'C:/Users/sherl/Desktop/login.txt').read_text(encoding='utf-8-sig')
creds={ip:(u,p.strip()) for ip,u,p in re.findall(r'IP:\s*(\d+\.\d+\.\d+\.\d+)\s+User:\s*(\S+)\s+Password:\s*([^\r\n]+)',raw)}
c=paramiko.SSHClient();c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ip='172.18.32.78';u,p=creds[ip]
c.connect(ip,username=u,password=p,timeout=12,look_for_keys=False,allow_agent=False)
cmd='''hostname
df -h
free -h
for p in /data5/shenglin.zeng /data/shenglin.zeng /home/shenglin.zeng /tmp/shenglin.zeng; do
 if test -d "$p"; then find "$p" -maxdepth 4 -type d \\( -name '*sd3*' -o -name '*sdxl*' -o -name '*env*' -o -name '*model*' \\) | head -50; fi
done
command -v python || true
command -v python3 || true
nvidia-smi --query-gpu=index,uuid,memory.free,utilization.gpu --format=csv,noheader,nounits
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader,nounits
'''
_,o,_=c.exec_command(cmd,timeout=30);o.channel.set_combine_stderr(True)
print(o.read().decode(errors='replace'));print('EXIT',o.channel.recv_exit_status());c.close()
