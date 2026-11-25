import os,json,time,hashlib,concurrent.futures,requests
from pathlib import Path
from huggingface_hub import get_hf_file_metadata,hf_hub_url
root=Path('/data5/shenglin.zeng/models/Wan2.1-T2V-1.3B')
for name in ['diffusion_pytorch_model.safetensors','models_t5_umt5-xxl-enc-bf16.pth']:
 url=hf_hub_url('Wan-AI/Wan2.1-T2V-1.3B',name,endpoint='https://hf-mirror.com')
 meta=get_hf_file_metadata(url,timeout=30)
 dest=root/name
 if dest.exists() and dest.stat().st_size==meta.size:
  print('EXISTS',name,meta.size,flush=True);continue
 partial=next(iter((root/'.cache/huggingface/download').glob('*.'+meta.etag+'.incomplete')),None)
 work=root/(name+'.segmented')
 if partial and not work.exists(): partial.rename(work)
 if not work.exists(): work.touch()
 start=work.stat().st_size
 print('START',name,'size',meta.size,'resume',start,flush=True)
 chunk=32*1024**2;fd=os.open(work,os.O_RDWR);completed=0;t0=time.time()
 def fetch(bounds):
  a,b=bounds
  for retry in range(4):
   try:
    with requests.get(meta.location,headers={'Range':f'bytes={a}-{b}'},stream=True,timeout=(15,90)) as response:
     response.raise_for_status()
     cr=response.headers.get('Content-Range','')
     assert response.status_code==206 and cr.startswith(f'bytes {a}-{b}/'),(response.status_code,cr)
     offset=a
     for buf in response.iter_content(1024**2):
      written=0
      while written<len(buf):written+=os.pwrite(fd,buf[written:],offset+written)
      offset+=len(buf)
     assert offset==b+1,(offset,b)
    return b-a+1
   except Exception as e:
    if retry==3:raise
    print('RETRY',a,str(e)[:180],flush=True);time.sleep(2)
 ranges=[(a,min(a+chunk,meta.size)-1) for a in range(start,meta.size,chunk)]
 try:
  with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
   for n in pool.map(fetch,ranges):
    completed+=n;print('PROGRESS',name,round((start+completed)/meta.size*100,1),round(completed/(time.time()-t0)/1024**2,2),'MiB/s',flush=True)
 finally:os.close(fd)
 sha=hashlib.sha256()
 with work.open('rb') as f:
  while buf:=f.read(16*1024**2):sha.update(buf)
 assert sha.hexdigest()==meta.etag,(sha.hexdigest(),meta.etag)
 work.rename(dest)
 print('VERIFIED',name,flush=True)
print('ALL_MODEL_FILES_COMPLETE',flush=True)
