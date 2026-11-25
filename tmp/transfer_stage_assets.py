import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,'/data5/shenglin.zeng/tools/eval_transfer_20261025')
import paramiko
req=json.loads(sys.stdin.readline());root=Path('/data5/shenglin.zeng/sdxl_crossdistill')
dest='/data1/shenglin.zeng/teacher_turning_20261026/stage_assets'
c=paramiko.SSHClient();c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('172.18.32.78',username=req['user'],password=req.pop('password'),timeout=15,look_for_keys=False,allow_agent=False)
c.get_transport().set_keepalive(20)
ck=root/'outputs/joint4_uniform_10k_2gpu_mb16_20261023/train/weights/20261024T063356+0800_g000160_periodic'
files=[(f,'checkpoint/'+f.name) for f in ck.iterdir() if f.name in ['student_rank0.pt','student_rank1.pt','shapes_rank0.json','complete.json']]
files+=[(ck.parent.parent/'config.json','checkpoint/config.json')]
files+=[(f,'dinov2-small/'+f.name) for f in (root/'models/dinov2-small').iterdir() if f.name in ['config.json','model.safetensors']]
assets=root/'exports/stage_probe_20261026'
files+=[(assets/'lpips_squeeze.pt','lpips_squeeze.pt')]
files+=[(f,'packages/'+str(f.relative_to(assets/'packages'))) for f in (assets/'packages').rglob('*') if f.is_file() and '__pycache__' not in str(f)]
manifest={}
with c.open_sftp() as s:
 def mkdir(path):
  try:s.stat(path)
  except OSError:
   mkdir(str(Path(path).parent));s.mkdir(path)
 mkdir(dest)
 for f,name in files:
  target=dest+'/'+name;mkdir(str(Path(target).parent))
  s.put(str(f),target+'.partial');s.posix_rename(target+'.partial',target)
  if not name.startswith('packages/'):
   h=hashlib.sha256()
   with f.open('rb') as inp:
    for b in iter(lambda:inp.read(8<<20),b''):h.update(b)
   manifest[name]={'bytes':f.stat().st_size,'sha256':h.hexdigest()};print('TRANSFER',name,f.stat().st_size,flush=True)
 with s.open(dest+'/manifest.json','w') as out:out.write(json.dumps(manifest,indent=2))
c.close();print('TRANSFER_DONE',flush=True)
