import sys, re, json
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / '.image_runtime_pkgs'))
import paramiko
raw=Path(r'C:\Users\sherl\Desktop\login.txt').read_text(encoding='utf-8-sig')
u,p=next((u,p.strip()) for ip,u,p in re.findall(r'IP:\s*(\d+\.\d+\.\d+\.\d+)\s+User:\s*(\S+)\s+Password:\s*([^\r\n]+)',raw) if ip=='172.18.132.41')
c=paramiko.SSHClient();c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('172.18.132.41',username=u,password=p,timeout=15,look_for_keys=False,allow_agent=False)
command='''id
ls -l /dev/nvidia[0-9]* /dev/nvidiactl /dev/nvidia-uvm
nvidia-smi -i 4,5,6,7 --query-gpu=index,pci.bus_id,driver_version,pstate,temperature.gpu,power.draw,compute_mode --format=csv
nvidia-smi -i 4 -q
cat /proc/driver/nvidia/gpus/*/information
dmesg --ctime 2>&1 | grep -iE 'NVRM|Xid|fallen|permission' | tail -20
ls -la /data1/shenglin.zeng
ls -d /home/shenglin.zeng/*conda* /data1/shenglin.zeng/*conda* /data1/shenglin.zeng/*env* 2>/dev/null
'''
_,out,_=c.exec_command(command,timeout=40);out.channel.set_combine_stderr(True)
result=out.read().decode(errors='replace');c.close()
Path('output/gpu_4090_health_20261022.txt').write_text(result,encoding='utf-8')
print(result)
