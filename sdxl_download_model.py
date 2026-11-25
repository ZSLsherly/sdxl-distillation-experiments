"""Official SDXL snapshot, independent resumable byte ranges and SHA256."""
import concurrent.futures,hashlib,json,time,requests
from pathlib import Path
from huggingface_hub import get_hf_file_metadata,hf_hub_url
root=Path('/data5/shenglin.zeng/sdxl_crossdistill')
info=json.loads((root/'model_info.json').read_text());repo=info['id'];revision=info['sha']
dest=root/'models/sdxl-base-1.0';dest.mkdir(parents=True,exist_ok=True)
prefixes={'scheduler','text_encoder','text_encoder_2','tokenizer','tokenizer_2','unet','vae'}
selected=[x['rfilename'] for x in info['siblings'] if x['rfilename']=='model_index.json' or (x['rfilename'].split('/')[0] in prefixes and (x['rfilename'].endswith('.json') or x['rfilename'].endswith('/merges.txt') or x['rfilename'].endswith('.fp16.safetensors')))]
manifest=[]
for name in selected:
 meta=get_hf_file_metadata(hf_hub_url(repo,name,revision=revision,endpoint='https://huggingface.co'),timeout=90)
 path=dest/name;path.parent.mkdir(parents=True,exist_ok=True)
 if path.exists() and path.stat().st_size==meta.size:
  print('EXISTS',name,flush=True);continue
 print('DOWNLOAD',name,meta.size,flush=True);started=time.time()
 if meta.size<8*1024**2:
  response=requests.get(meta.location,timeout=120);response.raise_for_status();path.write_bytes(response.content)
 else:
  parts=path.parent/(path.name+'.parts');parts.mkdir(exist_ok=True);chunk=32*1024**2
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
     with session.get(meta.location,headers={'Range':f'bytes={begin}-{b}'},stream=True,timeout=(15,120)) as response:
      response.raise_for_status();cr=response.headers.get('Content-Range','')
      assert response.status_code==206 and cr.startswith(f'bytes {begin}-{b}/'),(response.status_code,cr)
      with partial.open('ab') as out:
       for buf in response.iter_content(1024**2):out.write(buf)
     assert p.with_suffix('.tmp').stat().st_size==b-a+1
     p.with_suffix('.tmp').replace(p);return
    except Exception as e:
     print('RETRY',name,a,type(e).__name__,flush=True)
     if retry==5:raise
     time.sleep(2+retry)
  with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
   futs=[pool.submit(fetch,b) for b in bounds]
   for count,fut in enumerate(concurrent.futures.as_completed(futs),1):
    fut.result()
    if count%5==0 or count==len(bounds):print('PROGRESS',name,count,len(bounds),round(time.time()-started,1),flush=True)
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
print('SDXL_MODEL_DOWNLOAD_COMPLETE',flush=True)
