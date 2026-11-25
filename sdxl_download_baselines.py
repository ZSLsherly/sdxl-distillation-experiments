"""Official full UNet checkpoints for DMD2 and the 2026 SenseFlow baseline."""
import concurrent.futures,hashlib,json,time,requests
from pathlib import Path
from huggingface_hub import get_hf_file_metadata,hf_hub_url
root=Path('/data5/shenglin.zeng/sdxl_crossdistill/models/baselines');root.mkdir(parents=True,exist_ok=True)
tasks=[('tianweiy/DMD2','be22767697a1f3ca656b73c776e15fa335c86c6c','dmd2_sdxl_4step_unet_fp16.safetensors','dmd2/unet.safetensors'),
 ('domiso/SenseFlow',None,'SenseFlow-SDXL/unet/diffusion_pytorch_model.safetensors','senseflow/unet.safetensors')]
manifest=[]
for repo,revision,name,output in tasks:
 if revision is None:
  r=requests.get('https://huggingface.co/api/models/'+repo,timeout=45);r.raise_for_status();revision=r.json()['sha']
 meta=get_hf_file_metadata(hf_hub_url(repo,name,revision=revision,endpoint='https://huggingface.co'),timeout=90)
 path=root/output;path.parent.mkdir(parents=True,exist_ok=True)
 if path.exists() and path.stat().st_size==meta.size:print('EXISTS',output,flush=True);continue
 print('DOWNLOAD',output,meta.size,flush=True);started=time.time();chunk=32*1024**2
 parts=path.parent/(path.name+'.parts');parts.mkdir(exist_ok=True)
 bounds=[(a,min(a+chunk,meta.size)-1) for a in range(0,meta.size,chunk)]
 def fetch(bound):
  a,b=bound;p=parts/str(a)
  if p.exists() and p.stat().st_size==b-a+1:return
  for retry in range(8):
   try:
    tmp=p.with_suffix('.tmp');offset=tmp.stat().st_size if tmp.exists() else 0
    if offset==b-a+1:tmp.replace(p);return
    assert 0<=offset<b-a+1;begin=a+offset
    session=requests.Session();session.trust_env=(retry>=5)
    with session.get(meta.location,headers={'Range':f'bytes={begin}-{b}'},stream=True,timeout=(15,180)) as r:
     r.raise_for_status();assert r.status_code==206 and r.headers.get('Content-Range','').startswith(f'bytes {begin}-{b}/')
     with tmp.open('ab') as out:
      for buf in r.iter_content(1024**2):out.write(buf)
    assert tmp.stat().st_size==b-a+1;tmp.replace(p);return
   except Exception as e:
    print('RETRY',output,a,type(e).__name__,flush=True)
    if retry==7:raise
    time.sleep(2+retry)
 with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
  for count,fut in enumerate(concurrent.futures.as_completed([pool.submit(fetch,b) for b in bounds]),1):
   fut.result()
   if count%20==0 or count==len(bounds):print('PROGRESS',output,count,len(bounds),round(time.time()-started,1),flush=True)
 sha=hashlib.sha256()
 with path.with_suffix('.incomplete').open('wb') as out:
  for a,b in bounds:
   with (parts/str(a)).open('rb') as inp:
    while buf:=inp.read(4*1024**2):out.write(buf);sha.update(buf)
 assert sha.hexdigest()==meta.etag,(output,sha.hexdigest(),meta.etag)
 path.with_suffix('.incomplete').replace(path)
 for a,b in bounds:(parts/str(a)).unlink()
 parts.rmdir()
 manifest.append({'repo':repo,'revision':revision,'file':name,'size':meta.size,'sha256':sha.hexdigest()})
 print('VERIFIED',output,flush=True)
(root/'manifest.json').write_text(json.dumps(manifest,indent=2))
print('SDXL_BASELINES_DOWNLOAD_COMPLETE',flush=True)
