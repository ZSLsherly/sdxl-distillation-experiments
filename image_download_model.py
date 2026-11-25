"""Pinned official model download; resumable independent ranges, final SHA256."""
import concurrent.futures,hashlib,json,os,time,requests
from pathlib import Path
from huggingface_hub import get_hf_file_metadata,hf_hub_url
root=Path('/data5/shenglin.zeng/image_crossdistill')
info=json.loads((root/'model_info.json').read_text());repo=info['id'];revision=info['sha']
dest=root/'models/Sana_600M_512px_diffusers';dest.mkdir(parents=True,exist_ok=True)
files=[x['rfilename'] for x in info['siblings']]
selected=[f for f in files if f.endswith('.json') and 'fp16' not in f]
selected += ['tokenizer/tokenizer.model','transformer/diffusion_pytorch_model.fp16.safetensors',
 'text_encoder/model-00001-of-00002.safetensors','text_encoder/model-00002-of-00002.safetensors',
 'vae/diffusion_pytorch_model.safetensors']
manifest=[]
for name in selected:
 url=hf_hub_url(repo,name,revision=revision,endpoint='https://hf-mirror.com')
 meta=get_hf_file_metadata(url,timeout=60);path=dest/name;path.parent.mkdir(parents=True,exist_ok=True)
 if path.exists() and path.stat().st_size==meta.size:
  print('EXISTS',name,flush=True);continue
 print('DOWNLOAD',name,meta.size,flush=True);started=time.time()
 if meta.size<8*1024**2:
  response=requests.get(url,timeout=120);response.raise_for_status();path.write_bytes(response.content)
 else:
  parts=path.parent/(path.name+'.parts');parts.mkdir(exist_ok=True);chunk=32*1024**2
  bounds=[(a,min(a+chunk,meta.size)-1) for a in range(0,meta.size,chunk)]
  def fetch(bound):
   a,b=bound;p=parts/str(a)
   if p.exists() and p.stat().st_size==b-a+1:return
   for retry in range(6):
    try:
     with requests.get(meta.location,headers={'Range':f'bytes={a}-{b}'},stream=True,timeout=(30,120)) as response:
      response.raise_for_status();cr=response.headers.get('Content-Range','')
      assert response.status_code==206 and cr.startswith(f'bytes {a}-{b}/'),(response.status_code,cr)
      with p.with_suffix('.tmp').open('wb') as out:
       for buf in response.iter_content(1024**2):out.write(buf)
     assert p.with_suffix('.tmp').stat().st_size==b-a+1
     p.with_suffix('.tmp').replace(p);return
    except Exception as e:
     print('RETRY',name,a,str(e)[:100],flush=True)
     if retry==5:raise
     time.sleep(2+retry)
  with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
   futs=[pool.submit(fetch,b) for b in bounds]
   for count,fut in enumerate(concurrent.futures.as_completed(futs),1):
    fut.result()
    if count%10==0 or count==len(bounds):print('PROGRESS',name,count,len(bounds),round(time.time()-started,1),flush=True)
  sha=hashlib.sha256()
  with path.with_suffix(path.suffix+'.incomplete').open('wb') as out:
   for a,b in bounds:
    with (parts/str(a)).open('rb') as inp:
     while buf:=inp.read(4*1024**2):out.write(buf);sha.update(buf)
  assert sha.hexdigest()==meta.etag,(name,sha.hexdigest(),meta.etag)
  path.with_suffix(path.suffix+'.incomplete').replace(path)
  for a,b in bounds:(parts/str(a)).unlink()
  parts.rmdir()
 assert path.stat().st_size==meta.size
 manifest.append({'file':name,'size':meta.size,'etag':meta.etag})
 print('VERIFIED',name,round(time.time()-started,1),flush=True)
(root/'download_manifest.json').write_text(json.dumps({'repo':repo,'revision':revision,'files':manifest},indent=2))
print('MODEL_DOWNLOAD_COMPLETE',flush=True)
