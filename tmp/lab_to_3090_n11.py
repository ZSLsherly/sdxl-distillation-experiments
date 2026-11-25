import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,'/data5/shenglin.zeng/tools/eval_transfer_20261025')
import paramiko
req=json.loads(sys.stdin.readline());dst=paramiko.SSHClient();dst.set_missing_host_key_policy(paramiko.AutoAddPolicy())
dst.connect('172.18.32.78',username=req['user'],password=req.pop('password'),timeout=20,look_for_keys=False,allow_agent=False)
dst.get_transport().set_keepalive(20)
base=Path('/data5/shenglin.zeng');target=Path('/data1/shenglin.zeng/coco_missing_20261111')
archive=base/'checkpoints/a800_archive_20261109/outputs/A_pcm10_mix1_g160/model'
ref=base/'checkpoints/h20_gan_20261104/sdxl_runs/outputs/h20_C_mix1_terminal_GAN_G160_coco10k'
items=[(p,target/'checkpoints/A_pcm10_mix1'/p.relative_to(archive)) for p in archive.rglob('*') if p.is_file()]
model=base/'sdxl_crossdistill/models/sdxl-base-1.0'
items += [(p,target/'models/sdxl-base-1.0'/p.relative_to(model)) for p in model.rglob('*') if p.is_file()]
items += [(base/'sdxl_crossdistill/models/metrics/clean_fid/inception-2015-12-05.pt',target/'models/metrics/clean_fid/inception-2015-12-05.pt'),(ref/'manifest.json',target/'reference/manifest.json')]
items += [(p,target/'reference/features/real'/p.name) for p in (ref/'features/real').glob('*.npz')]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
with dst.open_sftp() as ftp:
 made=set();manifest={}
 for i,(src,dest) in enumerate(items):
  for par in reversed(dest.parents):
   if str(par) in made:continue
   try:ftp.mkdir(str(par))
   except IOError:ftp.stat(str(par))
   made.add(str(par))
  h=sha(src);manifest[str(dest.relative_to(target))]=[src.stat().st_size,h]
  try:existing=ftp.stat(str(dest)).st_size==src.stat().st_size
  except IOError:existing=False
  if not existing:ftp.put(str(src),str(dest)+'.partial');ftp.posix_rename(str(dest)+'.partial',str(dest))
  if i<25 or i%50==0:print('COPIED',i+1,len(items),str(dest.relative_to(target)),flush=True)
 with ftp.open(str(target/'transfer_manifest.json'),'w') as f:f.write(json.dumps(manifest))
print('TRANSFER_COMPLETE',len(items),flush=True)
