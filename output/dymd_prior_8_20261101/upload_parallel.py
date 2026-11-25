import sys,re,time,threading,queue,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'.image_runtime_pkgs'))
import paramiko
raw=Path(r'C:\Users\sherl\Desktop\login.txt').read_text(encoding='utf-8-sig')
creds={ip:(u,p.strip()) for ip,u,p in re.findall(r'IP:\s*(\d+\.\d+\.\d+\.\d+)\s+User:\s*(\S+)\s+Password:\s*([^\r\n]+)',raw)}
ip='172.18.132.41';u,pw=creds[ip]
root=Path(__file__).resolve().parent
remote='/data1/shenglin.zeng/dymd_prior_8_20261101/'
def connect():
 c=paramiko.SSHClient();c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
 c.connect(ip,username=u,password=pw,timeout=20,look_for_keys=False,allow_agent=False)
 return c
def upload(src,dest):
 size=src.stat().st_size;chunk=4*1024**2;tasks=queue.Queue();errors=[]
 state=root/(src.name+'.uploaded.json');done=set(json.loads(state.read_text())) if state.exists() else set()
 c=connect();s=c.open_sftp()
 try:
  exists=s.stat(dest).st_size==size
 except IOError:exists=False
 if exists:
  h=hashlib.sha256()
  with src.open('rb') as f:
   while x:=f.read(16*1024**2):h.update(x)
  _,o,_=c.exec_command("sha256sum '"+dest+"'",timeout=1800)
  if o.read().decode().split()[0]==h.hexdigest():
   s.close();c.close();print('ALREADY_VERIFIED',src.name,flush=True);return
 try:
  s.stat(dest+'.parallel')
 except IOError:
  with s.open(dest+'.parallel','w') as f:f.truncate(size)
  done=set()
 s.close();c.close()
 for a in range(0,size,chunk):
  if a not in done:tasks.put(a)
 lock=threading.Lock();start=time.time();total=[0]
 def worker():
  c=None
  try:
   c=connect();s=c.open_sftp()
   s.get_channel().settimeout(90)
   with src.open('rb') as f:
    while True:
     try:a=tasks.get_nowait()
     except queue.Empty:break
     f.seek(a);buf=f.read(min(chunk,size-a))
     with s.open(dest+'.parallel','r+') as g:
      g.set_pipelined(True);g.seek(a);g.write(buf)
     # Closing drains outstanding write acknowledgements before committing the chunk.
     with lock:
      done.add(a);total[0]+=len(buf);state.write_text(json.dumps(sorted(done)))
      if len(done)%16==0:print('UPLOAD',src.name,len(done)*4,'MiB',round(total[0]/(time.time()-start)/1024**2,2),'MiB/s',flush=True)
  except Exception as e:
   errors.append(e)
  finally:
   if c:c.close()
 threads=[]
 for _ in range(16):
  t=threading.Thread(target=worker);t.start();threads.append(t);time.sleep(.15)
 for t in threads:t.join()
 if errors:raise errors[0]
 assert len(done)==len(range(0,size,chunk))
 h=hashlib.sha256()
 with src.open('rb') as f:
  while x:=f.read(16*1024**2):h.update(x)
 c=connect()
 cmd="sha256sum '"+dest+".parallel'"
 _,out,_=c.exec_command(cmd,timeout=1800);actual=out.read().decode().split()[0]
 assert actual==h.hexdigest(),(src.name,actual,h.hexdigest())
 s=c.open_sftp();s.posix_rename(dest+'.parallel',dest);s.close();c.close()
 print('REMOTE_VERIFIED',src.name,actual,flush=True)

if __name__=='__main__':
 for arg in sys.argv[1:]:
  src=root/arg
  while not src.exists():time.sleep(10)
  dest=remote+('official/pf_wan/checkpoints/PF_Wan/' if arg.startswith('weights/') else 'wheels/')+src.name
  c=connect();s=c.open_sftp()
  try:s.mkdir(remote+'wheels')
  except IOError:pass
  s.close();c.close();upload(src,dest)
