import sys,json,shlex,subprocess,os
from pathlib import Path
from a50k_stage_assets_n23 import AUTH,remote,RCLONE,ENV
H='172.18.33.60'
R='/data5/shenglin.zeng/a50k_20261124'
S='/data/shenglin.zeng/a50k_20261123'
N='parallel_50k_seed20261018'
L=Path('output/sdxl-crossdistill/a50k_train60_20261124')
if sys.argv[1]=='pull':
 import paramiko
 c=paramiko.SSHClient();c.set_missing_host_key_policy(paramiko.AutoAddPolicy());c.connect(H,username=AUTH[H][0],password=AUTH[H][1],timeout=25,look_for_keys=False,allow_agent=False)
 py="""import os,sys,json,subprocess
r=json.loads(sys.stdin.readline());exe='/data5/shenglin.zeng/checkpoints/a800_archive_20261114/.tools/rclone'
env=dict(os.environ,RCLONE_CONFIG='/dev/null',RCLONE_CONFIG_STORE_TYPE='sftp',RCLONE_CONFIG_STORE_HOST='172.18.32.81',RCLONE_CONFIG_STORE_USER=r['u'],RCLONE_CONFIG_STORE_PASS=subprocess.check_output([exe,'obscure','-'],input=r.pop('p')+'\\n',universal_newlines=True).strip())
os.makedirs(r['dst'],exist_ok=True)
subprocess.run([exe,'copy','store:'+r['src'],r['dst'],'--max-depth','1','--transfers','3'],env=env,check=True)
import hashlib
from pathlib import Path
d=Path(r['dst']);v=json.loads((d/'validation.json').read_text());assert v['train']==50000 and v['holdout']==32
rows=[]
for f in sorted(d.iterdir()):
 if not f.is_file():continue
 h=subprocess.check_output(['sha256sum',str(f)]).decode().split()[0];ref=r['hashes'][f.name];assert h==ref
 if f.name=='prepared.pt':assert h==v['sha256']
 rows.append('172.18.32.81:'+r['src']+'/'+f.name+' -> 172.18.33.60:'+str(f)+' SHA256 '+h)
(d.parent/'已核验传输.txt').write_text('\\n'.join(rows)+'\\n');print('VERIFIED',rows,flush=True)
"""
 i,o,e=c.exec_command('python3 -u -c '+shlex.quote(py),timeout=7200);o.channel.set_combine_stderr(True);i.write(json.dumps(dict(u=AUTH['172.18.32.81'][0],p=AUTH['172.18.32.81'][1],src=S+'/datasets/'+N,dst=R+'/datasets/'+N,hashes={line.split('  ',1)[1]:line.split()[0] for line in remote('cd '+S+'/datasets/'+N+' && sha256sum -- *.*').splitlines()}))+'\n');i.flush();i.channel.shutdown_write()
 for line in o:print(line.rstrip(),flush=True)
 sys.exit(o.channel.recv_exit_status())
elif sys.argv[1]=='push':
 import hashlib
 folder=L/'deploy'; archive=L/'deploy.tar'
 import tarfile
 with tarfile.open(archive,'w') as tar:tar.add(folder,arcname='code')
 subprocess.run([RCLONE,'copyto',str(archive),'source:'+R+'/deploy.tar'],env=ENV,check=True)
 digest=hashlib.sha256(archive.read_bytes()).hexdigest()
 assert remote('sha256sum '+R+'/deploy.tar',H).split()[0]==digest
 (L/'部署校验.txt').write_text(str(archive.resolve())+' -> '+H+':'+R+'/deploy.tar SHA256 '+digest,encoding='utf-8')
 print(remote('cd '+R+' && tar xf deploy.tar',H))
