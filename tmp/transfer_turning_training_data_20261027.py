import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,'/data5/shenglin.zeng/tools/eval_transfer_20261025')
import paramiko
req=json.loads(sys.stdin.readline());source=Path('/data5/shenglin.zeng/sdxl_crossdistill/datasets/parallel_10k_seed20261018')
dest='/data1/shenglin.zeng/teacher_turning_20261026/datasets/parallel_10k_seed20261018'
c=paramiko.SSHClient();c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('172.18.32.78',username=req['user'],password=req.pop('password'),timeout=15,look_for_keys=False,allow_agent=False)
c.get_transport().set_keepalive(20);manifest={}
with c.open_sftp() as s:
 for path in ['/data1/shenglin.zeng/teacher_turning_20261026/datasets',dest]:
  try:s.mkdir(path)
  except OSError:s.stat(path)
 for name in ['prepared.pt','manifest.json']:
  f=source/name;h=hashlib.sha256()
  with f.open('rb') as inp:
   for b in iter(lambda:inp.read(8<<20),b''):h.update(b)
  expected={'bytes':f.stat().st_size,'sha256':h.hexdigest()};manifest[name]=expected
  last=[-1]
  def progress(done,total):
   n=done//(1<<30)
   if n!=last[0]:print('TRANSFER',name,round(done/2**30,2),'/',round(total/2**30,2),'GiB',flush=True);last[0]=n
  s.put(str(f),dest+'/'+name+'.partial',callback=progress);s.posix_rename(dest+'/'+name+'.partial',dest+'/'+name)
 with s.open(dest+'/transfer_checksums.json','w') as out:out.write(json.dumps(manifest,indent=2))
c.close();print('TRANSFER_DONE',json.dumps(manifest),flush=True)
