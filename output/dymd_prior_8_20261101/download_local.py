import concurrent.futures,hashlib,json,os,time,threading,faulthandler
from pathlib import Path
import requests
import traceback
from state_io import atomic_json,read_json
B=Path(__file__).resolve().parent/'weights';B.mkdir(exist_ok=True)
stall_trace=(B.parent/'download_stall.trace').open('a')
faulthandler.enable(file=stall_trace)
thread_sessions=threading.local()
def session():
 if not hasattr(thread_sessions,'http'):
  s=requests.Session()
  s.mount('https://',requests.adapters.HTTPAdapter(pool_connections=2,pool_maxsize=1))
  thread_sessions.http=s
 return thread_sessions.http
REV='8fa7e17d5f944d0275df7dc7929137b956367ff8'
files=[('Wan2.1_VAE.pth',507609880,'38071ab59bd94681c686fa51d75a1968f64e470262043be31f7a094e442fd981'),('backbone.pth',28578152777,'569e41d3046fe8bdd8df8345a9715e2d8fc5d92d52811ac165ad8947d776a47b'),('models_t5_umt5-xxl-enc-bf16.pth',11361920418,'7cace0da2b446bbbbc57d031ab6cf163a3d59b366da94e5afe36745b746fd81d')]
for name,size,sha in files:
 target=B/name;tmp=B/(name+'.partial');done=B/(name+'.chunks.json')
 if target.exists() and target.stat().st_size==size:continue
 resolve_url=f'https://huggingface.co/DAGroup-PKU/PF_Wan/resolve/{REV}/{name}'
 r=requests.head(resolve_url+'?download=true',headers={'Cache-Control':'no-cache'},allow_redirects=True,timeout=(15,30));r.raise_for_status();url=r.url
 url_lock=threading.Lock();url_time=time.time()
 completed=set(read_json(done)) if done.exists() else set()
 if not tmp.exists():
  with tmp.open('wb') as f:f.truncate(size)
 chunk=4*1024**2;start=time.time();total=0
 def fetch(a):
  global url,url_time
  z=min(a+chunk,size)-1
  for retry in range(10):
   try:
    if time.time()-url_time>1200:
     with url_lock:
      if time.time()-url_time>1200:
       refreshed=requests.head(resolve_url+'?download=true',headers={'Cache-Control':'no-cache'},allow_redirects=True,timeout=(15,30))
       refreshed.raise_for_status();url=refreshed.url;url_time=time.time()
       print('REFRESHED_DOWNLOAD_LINK',name,flush=True)
    with session().get(url,headers={'Range':f'bytes={a}-{z}'},stream=True,timeout=(15,45)) as q:
     if q.status_code in (401,403):
      with url_lock:
       if time.time()-url_time>5:
        refreshed=requests.head(resolve_url+'?download=true',headers={'Cache-Control':'no-cache'},allow_redirects=True,timeout=(15,30))
        refreshed.raise_for_status();url=refreshed.url;url_time=time.time()
     q.raise_for_status();assert q.status_code==206
     assert q.headers.get('Content-Range','').startswith(f'bytes {a}-{z}/')
     with tmp.open('r+b') as f:
      f.seek(a);n=0
      for buf in q.iter_content(1024**2):f.write(buf);n+=len(buf)
     assert n==z-a+1
    return a,n
   except Exception as e:
    if retry==9:raise
    detail=repr(getattr(e.args[0],'reason',None))[:200] if isinstance(e,requests.exceptions.SSLError) else ''
    print('RETRY',name,a,type(e).__name__,'attempt',retry+1,detail,flush=True);time.sleep(min(2**retry,15))
 pool=concurrent.futures.ThreadPoolExecutor(max_workers=8)
 try:
  faulthandler.dump_traceback_later(120,file=stall_trace)
  fs=[pool.submit(fetch,a) for a in range(0,size,chunk) if a not in completed]
  for f in concurrent.futures.as_completed(fs):
   a,n=f.result();completed.add(a);total+=n
   atomic_json(done,sorted(completed))
   faulthandler.dump_traceback_later(120,file=stall_trace)
   if len(completed)%16==0:print(name,'MiB',len(completed)*4,'speed',round(total/(time.time()-start)/1024**2,2),flush=True)
 except Exception:
  traceback.print_exc()
  pool.shutdown(wait=False,cancel_futures=True)
  # Leave uncommitted chunks for retry instead of joining thousands of queued requests.
  os._exit(1)
 else:pool.shutdown(wait=True)
 faulthandler.cancel_dump_traceback_later()
 h=hashlib.sha256()
 with tmp.open('rb') as f:
  while buf:=f.read(16*1024**2):h.update(buf)
 assert h.hexdigest()==sha,(name,h.hexdigest())
 for attempt in range(30):
  try:tmp.rename(target);break
  except PermissionError:
   if attempt==29:raise
   time.sleep(1)
 print('VERIFIED',name,flush=True)
