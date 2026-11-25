"""Use the installed FileZilla fzsftp binary; credentials stay only in memory."""
import os,sys,json,subprocess,selectors,time,hashlib,concurrent.futures,threading,tarfile
from pathlib import Path
BASE=Path('/data5/shenglin.zeng/checkpoints/a800_archive_20261106')
PRIVATE=BASE/'.pull6';ROOT='/root/autodl-tmp/sdxl_runs'
CONFIG=json.loads(sys.stdin.readline())
if os.fork():
 print('FILEZILLA_WORKER_STARTED',flush=True);sys.exit(0)
os.setsid()
log=os.open(str(PRIVATE/'filezilla.log'),os.O_WRONLY|os.O_CREAT|os.O_APPEND,0o600)
os.dup2(log,1);os.dup2(log,2)
os.dup2(os.open('/dev/null',os.O_RDONLY),0)
m=json.loads((BASE/'manifest.json').read_text())
assert not any(json.loads((BASE/'dedup_checked.json').read_text())['matches'].values())
class FZ:
 def __init__(self):
  env=os.environ.copy();env['HOME']=str(PRIVATE/'fzhome')
  env['LD_LIBRARY_PATH']='/data5/shenglin.zeng/tools/filezilla/app/usr/lib/x86_64-linux-gnu'
  self.p=subprocess.Popen(['/data5/shenglin.zeng/tools/filezilla/app/usr/bin/fzsftp'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,env=env)
  self.sel=selectors.DefaultSelector();self.sel.register(self.p.stdout,selectors.EVENT_READ)
  self.buf=b'';self.hostline=False;self.verified=False
  self.send('-0-');self.send('-1-');self.send('open "root@connect.nma1.seetacloud.com" 14892');self.wait()
 def send(self,line):
  self.p.stdin.write((line+'\n').encode());self.p.stdin.flush()
 def wait(self):
  last=time.monotonic()
  while True:
   if not self.buf:
    if not self.sel.select(3):
     if time.monotonic()-last>120:raise TimeoutError('FileZilla stalled')
     continue
    self.buf=os.read(self.p.stdout.fileno(),65536)
    if not self.buf:raise EOFError('FileZilla exited')
    last=time.monotonic()
   if self.buf[:1] in (b'6',b'7',b'@',b'A') and not self.hostline:
    typ=self.buf[:1];self.buf=self.buf[1:]
    if typ==b'@':self.send('-0-')
    if typ==b'A':self.send('-1-')
    continue
   if b'\n' not in self.buf:
    if not self.sel.select(3):
     if time.monotonic()-last>120:raise TimeoutError('FileZilla incomplete event')
     continue
    b=os.read(self.p.stdout.fileno(),65536)
    if not b:raise EOFError('FileZilla exited')
    self.buf+=b;last=time.monotonic();continue
   line,self.buf=self.buf.split(b'\n',1)
   if self.hostline:
    assert line.strip()==b'14892' and self.verified
    self.send('n');self.hostline=False;continue
   typ=line[:1];v=line[1:].decode(errors='replace')
   if typ==b'I':self.verified=CONFIG['sha256'] in v or CONFIG['md5'] in v
   elif typ in (b'9',b':'):
    assert 'connect.nma1.seetacloud.com' in v;self.hostline=True
   elif typ==b'<':
    assert self.verified;self.send(CONFIG['password'])
   elif typ==b'1':
    if v.strip()!='1':raise RuntimeError('FileZilla command failed: '+v)
    return
   elif typ==b'2':print('FILEZILLA_ERROR',v,flush=True)
 def get(self,src,dst):
  self.send(('reget' if dst.exists() and dst.stat().st_size else 'get')+' "'+src+'" "'+str(dst)+'"');self.wait()
 def close(self):
  if self.p.poll() is None:self.p.terminate()
  self.p.wait();self.sel.close()
def digest(f):
 h=hashlib.sha256()
 for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()
def check(p,n,h):
 assert p.stat().st_size==n,p
 with p.open('rb') as f:assert digest(f)==h,p
def transfer(task):
 kind,i,rel=task
 if kind=='large':
  dst=BASE/rel;src=ROOT+'/'+rel
 else:
  dst=BASE/'evidence'/f'part_{i:03d}.tar';src=ROOT+'/tmp/side_archive_n6/groups/'+dst.name
 dst.parent.mkdir(parents=True,exist_ok=True);part=dst.with_suffix(dst.suffix+'.fzpart')
 for attempt in range(4):
  fz=None
  try:
   if not dst.exists():
    fz=FZ();fz.get(src,part);fz.close();fz=None
   target=dst if dst.exists() else part
   if kind=='large':check(target,*m['records'][rel])
   else:
    with tarfile.open(target,mode='r|') as tar:
     seen=[]
     for a in tar:
      assert a.isfile() and a.name in m['groups'][i]
      n,h=m['records'][a.name];assert a.size==n and digest(tar.extractfile(a))==h
      seen.append(a.name)
     assert seen==m['groups'][i]
   if not dst.exists():os.link(part,dst);part.unlink()
   if kind=='large':
    with (BASE/'verified_large.txt').open('a') as f:f.write(m['records'][rel][1]+'  '+rel+'\n')
    old=dst.with_suffix('.direct-part')
    if old.is_file():old.unlink()
   print('VERIFIED_FILEZILLA',kind,rel,flush=True);return
  except Exception as exc:
   print('RETRY_FILEZILLA',kind,rel,attempt,type(exc).__name__,str(exc)[:200],flush=True)
   if fz:fz.close()
   if isinstance(exc,AssertionError) and part.exists():part.unlink()
   if attempt==3:raise
   time.sleep(5)
tasks=[('large',i,r) for i,r in enumerate(m['large'])]+[('group',i,str(i)) for i in range(len(m['groups']))]
print('FILEZILLA_BEGIN',len(tasks),'transfers; 5 parallel connections',flush=True)
with concurrent.futures.ThreadPoolExecutor(5) as pool:
 for _ in pool.map(transfer,tasks):pass
(BASE/'COMPLETE.json').write_text(json.dumps({'files':len(m['records']),'bytes':sum(r[0] for r in m['records'].values()),'large':len(m['large']),'tar_groups':len(m['groups']),'transport':'FileZilla fzsftp'}))
print('FILEZILLA_COMPLETE',flush=True)
