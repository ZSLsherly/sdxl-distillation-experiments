"""Run on .33.60: copy only evaluation inputs directly between lab servers."""
import sys,json,hashlib,tarfile,io,posixpath,os
from pathlib import Path
sys.path.insert(0,'/data5/shenglin.zeng/tools/eval_transfer_20261025')
import paramiko
auth=json.loads(sys.stdin.readline())
if '--detach' in sys.argv:
 pid=os.fork()
 if pid:
  print('STAGING_PID',pid,flush=True);sys.exit(0)
 os.setsid()
 log=open('/data5/shenglin.zeng/tools/eval_transfer_20261025/clip_all_stage_20261113.log','a',buffering=1)
 os.dup2(log.fileno(),1);os.dup2(log.fileno(),2)
 with open('/dev/null') as null:os.dup2(null.fileno(),0)
def conn(host):
 c=paramiko.SSHClient();c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
 c.connect(host,username=auth[host][0],password=auth[host][1],look_for_keys=False,allow_agent=False,timeout=20)
 c.get_transport().set_keepalive(20);return c
dst=conn('172.18.32.81');ftp=dst.open_sftp();lab=conn('172.18.32.78');old=lab.open_sftp()
target='/data3/shenglin.zeng/clip_missing_20261113'
b=Path('/data5/shenglin.zeng');made=set()
def mkdir(p):
 if p in made:return
 par=posixpath.dirname(p)
 if par and par!=p:mkdir(par)
 try:ftp.mkdir(p)
 except IOError:ftp.stat(p)
 made.add(p)
def putbytes(data,dest):
 mkdir(posixpath.dirname(dest))
 with ftp.open(dest+'.partial','wb') as f:f.write(data)
 ftp.posix_rename(dest+'.partial',dest)
def readjson(p,remote=False):
 if remote:
  with old.open(str(p)) as f:return json.load(f)
 return json.loads(Path(p).read_text())
def same_samples(a,b):
 # source_index was added by newer manifests; it is not an evaluation input.
 keys=('index','prompt','seed','real_file')
 return len(a)==len(b) and all(tuple(x[k] for k in keys)==tuple(y[k] for k in keys) for x,y in zip(a,b))
def copy(src,dest,remote=False,expected=None):
 mkdir(posixpath.dirname(dest));h=hashlib.sha256();size=0
 # Resume preparation without copying verified destination files again.
 try:
  size0=old.stat(str(src)).st_size if remote else Path(src).stat().st_size
  if ftp.stat(dest).st_size==size0:
   source_sha=expected
   if source_sha is None and not remote:
    with open(src,'rb') as f:source_sha=hashlib.file_digest(f,'sha256').hexdigest()
   if source_sha:
    _,o,_=dst.exec_command('sha256sum '+dest)
    if o.read().decode().split()[0]==source_sha and o.channel.recv_exit_status()==0:
     return dict(name=posixpath.basename(dest),bytes=size0,sha256=source_sha,source=str(src))
 except FileNotFoundError:pass
 source=old.open(str(src),'rb') if remote else open(src,'rb')
 if remote:source.prefetch(old.stat(str(src)).st_size,max_concurrent_requests=32)
 with source as f,ftp.open(dest+'.partial','wb') as g:
  g.set_pipelined(True)
  while True:
   z=f.read(8<<20)
   if not z:break
   h.update(z);size+=len(z);g.write(z)
 actual=h.hexdigest()
 if expected and actual!=expected:raise ValueError('source checksum mismatch '+str(src))
 ftp.posix_rename(dest+'.partial',dest)
 _,o,e=dst.exec_command('sha256sum '+dest)
 got=o.read().decode().split()[0]
 assert o.channel.recv_exit_status()==0 and got==actual,(dest,got,actual)
 
 if size>10_000_000:print('COPIED',dest,size,flush=True)
 return dict(name=posixpath.basename(dest),bytes=size,sha256=actual,source=str(src))
refpath=b/'sdxl_crossdistill/outputs/coco10k_g160_20261020/manifest.json'
ref=readjson(refpath)
def stage(arm,rec,weightdir,images=None,remote=False,shape=None):
 out=target+'/arms/'+arm
 try:ftp.stat(out+'/ready.json');print('READY_EXISTS',arm,flush=True);return
 except IOError:pass
 ws=rec['world_size'];shards=[];expected={x['name']:x.get('sha256') for x in rec.get('shards',[])}
 for i in range(ws):
  name=f'student_rank{i}.pt'
  shards.append(copy(str(weightdir)+'/'+name,out+'/weights/'+name,remote,expected.get(name)))
 if shape is not None:putbytes(shape,out+'/weights/shapes_rank0.json')
 else:
  src=str(weightdir)+'/shapes_rank0.json'
  try:copy(src,out+'/weights/shapes_rank0.json',remote)
  except FileNotFoundError:
   if ws!=1:raise
 rec=dict(rec,checkpoint=out+'/weights',shards=shards)
 manifest={k:ref[k] for k in ('samples','visual_indices','protocol','n')}
 manifest['models']={arm:rec}
 putbytes(json.dumps(manifest).encode(),out+'/manifest.json')
 count=0
 if images:
  for idx in ref['visual_indices']:
   p=str(images)+f'/{idx:05d}.png'
   try:copy(p,out+f'/images/{idx:05d}.png',remote);count+=1
   except FileNotFoundError:pass
 putbytes(json.dumps(dict(arm=arm,shards=shards,cached_images=count)).encode(),out+'/ready.json')
 print('ARM_READY',arm,count,flush=True)
if '--remaining-only' not in sys.argv:
 base_model=b/'sdxl_crossdistill/models/sdxl-base-1.0'
 for p in base_model.rglob('*'):
  if p.is_file():copy(p,target+'/models/sdxl-base-1.0/'+str(p.relative_to(base_model)))
 cliphost=conn('172.18.132.41');cf=cliphost.open_sftp()
 cliproot='/data1/shenglin.zeng/sdxl_eval_uniform_20261025/outputs/semantic_joint4_uniform_1024/model'
 for entry in cf.listdir_attr(cliproot):
  if entry.st_mode & 0o170000 == 0o100000:
   with cf.open(cliproot+'/'+entry.filename,'rb') as f:
    f.prefetch(entry.st_size,max_concurrent_requests=32)
    mkdir(target+'/models/clip');dest=target+'/models/clip/'+entry.filename
    with ftp.open(dest+'.partial','wb') as g:
     g.set_pipelined(True)
     while True:
      chunk=f.read(8<<20)
      if not chunk:break
      g.write(chunk)
    ftp.posix_rename(dest+'.partial',dest)
 # Copy known reference PNGs for scoring replication, not new diffusion generation.
 cr='/data1/shenglin.zeng/sdxl_eval_uniform_20261025/outputs/coco10k/images/joint4_uniform'
 for idx in ref['visual_indices']:
  with cf.open(cr+f'/{idx:05d}.png','rb') as f:
   f.prefetch(cf.stat(cr+f'/{idx:05d}.png').st_size,max_concurrent_requests=32)
   putbytes(f.read(),target+f'/validation_images/{idx:05d}.png')
 cf.close();cliphost.close()
 putbytes(b'{"complete":true}',target+'/assets_ready.json')
 for arm in ('dmd_only','gan4','joint4','joint34'):
  rec=ref['models'][arm];stage(arm,rec,rec['checkpoint'],refpath.parent/'images'/arm)
 for arm,run in [('DM234_fake3','c_mix1_dm234_fake3_20261103'),('DM1_mid4_fake3','c_mix1_first_dm_fake3_g160'),('DM1_uniform','dm1_mid4_uniform_mix1_g160')]:
  root=b/'checkpoints/a800_archive_20261105/outputs'/run/'model'
  paths=list((root/'weights').glob('*g000160*'));assert len(paths)==1
  stage(arm,dict(world_size=1,config=readjson(root/'config.json')),paths[0])
 root=b/'checkpoints/h20_gan_20261104';ev=root/'sdxl_runs/outputs/h20_C_mix1_terminal_GAN_G160_coco10k'
 m=readjson(ev/'manifest.json');keys=[k for k,v in m['models'].items() if 'checkpoint' in v];assert len(keys)==1,keys
 rec=m['models'][keys[0]];weightdir=root
 shapes=list((root/'sdxl_runs/outputs/h20_C_mix1_terminal_GAN_G160/weights').glob('*/shapes_rank0.json'));assert len(shapes)==1
 stage('H20_extra_GAN',rec,weightdir,ev/'images'/keys[0],shape=shapes[0].read_bytes())
# The two November 6 configs and saved COCO images live in verified evidence tar files.
for arm,run in [('A_pcm10_patchgan','A_pcm10_terminal_patchgan'),('A_mid_clean_x0','A_mid_clean_x0')]:
 assets={}
 prefix='outputs/'+run+'/'
 for tar in sorted((b/'checkpoints/a800_archive_20261106/evidence').glob('*.tar')):
  with tarfile.open(tar) as t:
   for member in t:
    name=member.name
    if name.startswith(prefix) and (name.endswith('/manifest.json') or name.endswith('/model/config.json') or ('/images/' in name and name.endswith('.png'))):
     assets[name]=t.extractfile(member).read()
 manifests=[json.loads(v) for k,v in assets.items() if k.endswith('/manifest.json') and 'samples' in json.loads(v)]
 assert len(manifests)==1
 m=manifests[0];assert same_samples(m['samples'],ref['samples']) and m['visual_indices']==ref['visual_indices']
 rec=m['models'][arm];out=target+'/arms/'+arm
 # Reuse archived images without extracting unrelated backup contents.
 for k,v in assets.items():
  if '/images/'+arm+'/' in k:putbytes(v,out+'/images/'+posixpath.basename(k))
 root=b/'checkpoints/a800_archive_20261109/outputs'/run/'model'
 paths=list((root/'weights').glob('*g000160*'));assert len(paths)==1
 stage(arm,rec,paths[0])
for arm in ('A_pcm10_mix1','normalized_u_g160','raw_angle_g160'):
 r='/data1/shenglin.zeng/coco_missing_20261111/results/'+arm
 m=readjson(r+'/manifest.json',True)
 assert same_samples(m['samples'],ref['samples']) and m['visual_indices']==ref['visual_indices']
 rec=m['models'][arm]
 stage(arm,rec,rec['checkpoint'],r+'/images/'+arm,True)
putbytes(b'{"complete":true}',target+'/staging_complete.json')
print('STAGING_COMPLETE',flush=True)
