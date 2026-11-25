import sys,re,json
from pathlib import Path
sys.path.insert(0,'.image_runtime_pkgs')
import paramiko
raw=Path(r'C:\Users\sherl\Desktop\login.txt').read_text(encoding='utf-8-sig')
creds={ip:(u,p.strip()) for ip,u,p in re.findall(r'IP:\s*(\d+\.\d+\.\d+\.\d+)\s+User:\s*(\S+)\s+Password:\s*([^\r\n]+)',raw)}
c=paramiko.SSHClient();c.set_missing_host_key_policy(paramiko.AutoAddPolicy());u,p=creds['172.18.33.60'];c.connect('172.18.33.60',username=u,password=p,timeout=15,look_for_keys=False,allow_agent=False)
root='/data5/shenglin.zeng/sdxl_crossdistill/outputs/hparam_pilot/teacher_stages'
out=Path('output/sdxl-crossdistill/hparam_pilot/teacher_stages');out.mkdir(parents=True,exist_ok=True)
with c.open_sftp() as s:
 for rank in range(4):
  f=f'rank{rank}.json';s.get(root+'/'+f,str(out/f))
  for seed in [10,11,12,13]:
   for t in [899,749,599,449]:
    f=f'prompt{rank}_seed{seed}_t{t}_x0.png';s.get(root+'/'+f,str(out/f))
   f=f'prompt{rank}_seed{seed}_final.png';s.get(root+'/'+f,str(out/f))
  print('DOWNLOADED_PROMPT',rank,flush=True)
c.close()
