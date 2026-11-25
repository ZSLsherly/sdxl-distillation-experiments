import sys,re
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'.image_runtime_pkgs'))
import paramiko
raw=Path(r'C:\Users\sherl\Desktop\login.txt').read_text(encoding='utf-8-sig')
creds={ip:(u,p.strip()) for ip,u,p in re.findall(r'IP:\s*(\d+\.\d+\.\d+\.\d+)\s+User:\s*(\S+)\s+Password:\s*([^\r\n]+)',raw)}
out=Path('output/sdxl-crossdistill/coco_uniform_g160_20261025')
for host in ('172.18.33.60','172.18.132.41'):
 c=paramiko.SSHClient();c.set_missing_host_key_policy(paramiko.AutoAddPolicy());u,p=creds[host]
 c.connect(host,username=u,password=p,timeout=20,look_for_keys=False,allow_agent=False)
 with c.open_sftp() as s:
  if host.endswith('.60'):
   for name in ('scores.json','complete.json'):s.get('/data5/shenglin.zeng/sdxl_crossdistill/outputs/coco10k_g160_20261020/'+name,str(out/('previous_'+name)))
  else:
   for name in ('guard_coco_eval_20261025.py','setup_eval_dest_20261025.sh'):s.put('tmp/'+name,'/data1/shenglin.zeng/sdxl_eval_uniform_20261025/'+name)
 c.close();print('SYNC_OK',host)
