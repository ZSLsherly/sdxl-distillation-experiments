import sys,os,re,time,hashlib,shlex,threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'.image_runtime_pkgs'))
import paramiko
pw=sys.stdin.readline().strip()
labroot='/data5/shenglin.zeng/checkpoints/h20_gan_20261104'
cloudroot='/root/autodl-tmp/'
weight='sdxl_runs/outputs/h20_C_mix1_terminal_GAN_G160/weights/20261104T112222+0800_g000160_end/student_rank0.pt'
items=[('student_rank0.pt',weight,'361ad217704fe6b44d01242ea7b91e6ce4c63c90090e40197da99edd0266e068'),('experiments.tar','h20_backup_20261104/experiments.tar','1678224369f6c966138ae793d1eec330025579796a5a48201bc0e4bd4eb6df12')]
local=Path('output/sdxl-crossdistill/h20_interruption_20261104')
raw=Path('C:/Users/sherl/Desktop/login.txt').read_text(encoding='utf-8-sig')
cred=re.search(r'IP:\s*172\.18\.33\.60\s+User:\s*(\S+)\s+Password:\s*([^\r\n]+)',raw)
def connect(host,port,user,password):
 c=paramiko.SSHClient();c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
 c.connect(host,port,username=user,password=password,timeout=20,look_for_keys=False,allow_agent=False)
 c.get_transport().set_keepalive(20);return c
def cloud():return connect('region-42.seetacloud.com',48852,'root',pw)
def lab():return connect('172.18.33.60',22,cred[1],cred[2].strip())
def run(c,cmd):
 _,o,e=c.exec_command(cmd); data=o.read()
 if o.channel.recv_exit_status():raise RuntimeError(e.read()[:400])
 return data
step=16<<20; tl=threading.local();start=time.time()
with cloud() as src,lab() as dst:
 for name,remote,digest in items[1:]:
  size=int(run(src,'stat -c %s '+cloudroot+remote))
  dest=labroot+'/'+name+'.partial';loc=local/(name+'.partial')
  assert not (local/name).exists()
  run(dst,'umask 077; test ! -e '+labroot+'/'+name+' && truncate -s '+str(size)+' '+dest)
  with loc.open('wb') as f:f.truncate(size)
  def copy(offset):
   length=min(step,size-offset)
   for tries in range(3):
    try:
     if not hasattr(tl,'src'):tl.src=cloud()
     if not hasattr(tl,'dst'):tl.dst=lab()
     buf=run(tl.src,f'dd if={cloudroot+remote} iflag=fullblock,skip_bytes,count_bytes bs=1M skip={offset} count={length} status=none')
     assert len(buf)==length
     with loc.open('r+b',buffering=0) as f:f.seek(offset);f.write(buf)
     ch=tl.dst.get_transport().open_session();ch.settimeout(240)
     ch.exec_command(f'dd of={dest} bs=1M oflag=seek_bytes seek={offset} conv=notrunc status=none')
     ch.sendall(buf);ch.shutdown_write()
     if ch.recv_exit_status():raise RuntimeError('lab write failed')
     ch.close();return length
    except Exception:
     for a in ('src','dst'):
      c=getattr(tl,a,None)
      if c:c.close();delattr(tl,a)
     if tries==2:raise
  total=0
  with ThreadPoolExecutor(max_workers=6) as pool:
   for i,n in enumerate(pool.map(copy,range(0,size,step))):
    total+=n
    if i%16==0 or total==size:print(name,total,size,round(time.time()-start),flush=True)
  with loc.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==digest
  dst=lab()
  assert run(dst,'sha256sum '+dest).decode().split()[0]==digest
  run(dst,'mv '+dest+' '+labroot+'/'+name);loc.replace(local/name)
  print('VERIFIED_LOCAL_AND_LAB',name,size,flush=True)

