"""Download official DMD2 data: LAION prompts, one real latent shard, COCO10K."""
import concurrent.futures,hashlib,json,time,requests
from pathlib import Path
from huggingface_hub import get_hf_file_metadata,hf_hub_url
root=Path('/data5/shenglin.zeng/sdxl_crossdistill/datasets/dmd2');root.mkdir(parents=True,exist_ok=True)
repo='tianweiy/DMD2';revision='be22767697a1f3ca656b73c776e15fa335c86c6c'
files=['data/laion/captions_laion_score6.25.pkl','data/coco/captions_coco14_test.pkl',
 'data/laion_vae_latents/sdxl_vae_latents_laion_500k/vae_latents_000.pt','data/coco/coco10k.zip']
manifest=[]
for name in files:
 meta=get_hf_file_metadata(hf_hub_url(repo,name,revision=revision,endpoint='https://huggingface.co'),timeout=90)
 path=root/Path(name).name
 if path.exists() and path.stat().st_size==meta.size:print('EXISTS',name,flush=True);continue
 print('DOWNLOAD',name,meta.size,flush=True);started=time.time()
 parts=root/(path.name+'.parts');parts.mkdir(exist_ok=True);chunk=32*1024**2
 bounds=[(a,min(a+chunk,meta.size)-1) for a in range(0,meta.size,chunk)]
 def fetch(bound):
  a,b=bound;p=parts/str(a)
  if p.exists() and p.stat().st_size==b-a+1:return
  for retry in range(6):
   try:
    partial=p.with_suffix('.tmp');offset=partial.stat().st_size if partial.exists() else 0
    if offset==b-a+1:partial.replace(p);return
    assert 0<=offset<b-a+1
    begin=a+offset
    session=requests.Session();session.trust_env=(retry>=3)
    with session.get(meta.location,headers={'Range':f'bytes={begin}-{b}'},stream=True,timeout=(15,180)) as response:
     response.raise_for_status();cr=response.headers.get('Content-Range','')
     if meta.size<=chunk and response.status_code==200:assert offset==0 and int(response.headers['Content-Length'])==meta.size
     else:assert response.status_code==206 and cr.startswith(f'bytes {begin}-{b}/'),(response.status_code,cr)
     with partial.open('ab') as out:
      for buf in response.iter_content(1024**2):out.write(buf)
    assert p.with_suffix('.tmp').stat().st_size==b-a+1
    p.with_suffix('.tmp').replace(p);return
   except Exception as e:
    print('RETRY',name,a,type(e).__name__,flush=True)
    if retry==5:raise
    time.sleep(2+retry)
 with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
  futs=[pool.submit(fetch,b) for b in bounds]
  for count,fut in enumerate(concurrent.futures.as_completed(futs),1):
   fut.result()
   if count%5==0 or count==len(bounds):print('PROGRESS',name,count,len(bounds),round(time.time()-started,1),flush=True)
 sha=hashlib.sha256()
 with path.with_suffix(path.suffix+'.incomplete').open('wb') as out:
  for a,b in bounds:
   with (parts/str(a)).open('rb') as inp:
    while buf:=inp.read(4*1024**2):out.write(buf);sha.update(buf)
 if len(meta.etag)==64:assert sha.hexdigest()==meta.etag,(name,sha.hexdigest(),meta.etag)
 path.with_suffix(path.suffix+'.incomplete').replace(path)
 for a,b in bounds:(parts/str(a)).unlink()
 parts.rmdir()
 manifest.append({'repo':repo,'revision':revision,'file':name,'size':meta.size,'sha256':sha.hexdigest(),'etag':meta.etag})
 print('VERIFIED',name,round(time.time()-started,1),flush=True)
(root/'download_manifest.json').write_text(json.dumps(manifest,indent=2))
print('SDXL_DATA_DOWNLOAD_COMPLETE',flush=True)
