"""Remove only the two verified, disposable transport archives created today."""
import sys,re,json
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'.image_runtime_pkgs'))
import paramiko
raw=Path(r'C:\Users\sherl\Desktop\login.txt').read_text(encoding='utf-8-sig')
creds={ip:(u,p.strip()) for ip,u,p in re.findall(r'IP:\s*(\d+\.\d+\.\d+\.\d+)\s+User:\s*(\S+)\s+Password:\s*([^\r\n]+)',raw)}
proof=json.loads(Path('output/sdxl-crossdistill/joint34_uniform_20261025/design.json').read_text())
assert proof['training_data_sha256']=='fb390ee46c0fa9c39ae6e73f0d72a0a589163c938890615cd2d68a8c4ca2572a'
for host,folder in [('172.18.33.60','/data5/shenglin.zeng/sdxl_crossdistill/exports/joint34_uniform_20261025'),('172.18.132.41','/data1/shenglin.zeng/sdxl_eval_uniform_20261025/joint34_transfers')]:
 c=paramiko.SSHClient();c.set_missing_host_key_policy(paramiko.AutoAddPolicy());u,p=creds[host]
 c.connect(host,username=u,password=p,timeout=20,look_for_keys=False,allow_agent=False)
 with c.open_sftp() as s:
  with s.open(folder+'/transfer_manifest.json') as f:manifest=json.loads(f.read())
  for name in ('training_assets.tar',):
   assert s.stat(folder+'/'+name).st_size==manifest[name]['bytes']
   s.remove(folder+'/'+name);print('REMOVED_TRANSPORT_ARCHIVE',host,name,manifest[name]['bytes'])
 c.close()
