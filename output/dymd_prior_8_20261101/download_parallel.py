import concurrent.futures,hashlib,json,os,time
from pathlib import Path
import requests
from huggingface_hub import get_hf_file_metadata,hf_hub_url,hf_hub_download
B=Path('/data1/shenglin.zeng/dymd_prior_8_20261101')
D=B/'official/pf_wan/checkpoints/PF_Wan'
REV='8fa7e17d5f944d0275df7dc7929137b956367ff8'
repo='DAGroup-PKU/PF_Wan'
chunk=8*1024**2
for name in ['Wan2.1_VAE.pth','models_t5_umt5-xxl-enc-bf16.pth','backbone.pth']:
 meta=get_hf_file_metadata(hf_hub_url(repo,name,revision=REV),timeout=60)
 target=D/name; tmp=D/(name+'.ranges'); done=D/(name+'.chunks.json')
 if target.exists() and target.stat().st_size==meta.size:
  print('EXISTS',name,flush=True);continue
 completed=set(json.loads(done.read_text())) if done.exists() else set()
 fd=os.open(tmp,os.O_CREAT|os.O_RDWR,0o644);os.ftruncate(fd,meta.size)
 started=time.time();total=0
 def fetch(a):
  z=min(a+chunk,meta.size)-1
  for attempt in range(8):
   try:
    with requests.get(meta.location,headers={'Range':f'bytes={a}-{z}'},stream=True,timeout=(20,120)) as r:
     r.raise_for_status();assert r.status_code==206,(r.status_code,name)
     assert r.headers['Content-Range'].startswith(f'bytes {a}-{z}/')
     pos=a
     for buf in r.iter_content(1024**2):
      off=0
      while off<len(buf):off+=os.pwrite(fd,buf[off:],pos+off)
      pos+=len(buf)
     assert pos==z+1
    return a,z-a+1
   except Exception as e:
    if attempt==7:raise
    print('RETRY',name,a,type(e).__name__,flush=True);time.sleep(min(2**attempt,30))
 try:
  with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
   futures=[pool.submit(fetch,a) for a in range(0,meta.size,chunk) if a not in completed]
   for f in concurrent.futures.as_completed(futures):
    a,n=f.result();completed.add(a);total+=n
    done.write_text(json.dumps(sorted(completed)))
    print('PROGRESS',name,len(completed),len(range(0,meta.size,chunk)),round(total/(time.time()-started)/1024**2,2),'MiB/s',flush=True)
 finally:os.close(fd)
 h=hashlib.sha256()
 with tmp.open('rb') as f:
  while buf:=f.read(16*1024**2):h.update(buf)
 assert h.hexdigest()==meta.etag,(name,h.hexdigest(),meta.etag)
 tmp.rename(target);print('SHA256_VERIFIED',name,meta.etag,flush=True)
for x in json.loads((B/'model_api.json').read_text())['siblings']:
 if not x['rfilename'].endswith('.pth'):
  hf_hub_download(repo,x['rfilename'],revision=REV,local_dir=D)
(B/'MODEL_READY').touch()
print('ALL_MODELS_READY',flush=True)
