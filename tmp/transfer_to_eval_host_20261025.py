"""Private server-to-server SFTP. Destination authentication arrives via SSH stdin."""
import sys,json,time
from pathlib import Path
sys.path.insert(0,'/data5/shenglin.zeng/tools/eval_transfer_20261025')
import paramiko
request=json.loads(sys.stdin.readline())
source=Path('/data5/shenglin.zeng/sdxl_crossdistill/exports/coco_uniform_g160_20261025')
manifest=json.loads((source/'transfer_manifest.json').read_text())
client=paramiko.SSHClient();client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('172.18.132.41',username=request['user'],password=request.pop('password'),
               timeout=20,look_for_keys=False,allow_agent=False)
client.get_transport().set_keepalive(20)
dest='/data1/shenglin.zeng/sdxl_eval_uniform_20261025/transfers'
with client.open_sftp() as sftp:
 try:sftp.mkdir(dest)
 except OSError:sftp.stat(dest)
 for name,meta in manifest.items():
  last=[-1]
  def progress(done,total):
   gib=done//(1<<30)
   if gib!=last[0]:print('TRANSFER',name,round(done/2**30,2),'/',round(total/2**30,2),'GiB',flush=True);last[0]=gib
  sftp.put(str(source/name),dest+'/'+name+'.partial',callback=progress)
  assert sftp.stat(dest+'/'+name+'.partial').st_size==meta['bytes']
  sftp.posix_rename(dest+'/'+name+'.partial',dest+'/'+name)
 sftp.put(str(source/'transfer_manifest.json'),dest+'/transfer_manifest.json')
client.close();print('TRANSFER_FINISHED',flush=True)
