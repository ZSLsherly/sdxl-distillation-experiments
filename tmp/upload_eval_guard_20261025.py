import sys,re
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'.image_runtime_pkgs'))
import paramiko
raw=Path(r'C:\Users\sherl\Desktop\login.txt').read_text(encoding='utf-8-sig')
u,p=next((u,p.strip()) for ip,u,p in re.findall(r'IP:\s*(\d+\.\d+\.\d+\.\d+)\s+User:\s*(\S+)\s+Password:\s*([^\r\n]+)',raw) if ip=='172.18.132.41')
c=paramiko.SSHClient();c.set_missing_host_key_policy(paramiko.AutoAddPolicy());c.connect('172.18.132.41',username=u,password=p,timeout=20,look_for_keys=False,allow_agent=False)
with c.open_sftp() as s:s.put('tmp/guard_coco_eval_20261025.py','/data1/shenglin.zeng/sdxl_eval_uniform_20261025/guard_coco_eval_20261025.py')
c.close();print('UPLOADED_EVALUATION_GUARD')
