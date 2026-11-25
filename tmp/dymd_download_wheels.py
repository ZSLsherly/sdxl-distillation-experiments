import requests,json,concurrent.futures,time,hashlib
from pathlib import Path
from urllib.parse import urlsplit,unquote
B=Path('output/dymd_prior_8_20261101/wheels');B.mkdir(exist_ok=True)
urls=json.loads(Path('output/dymd_prior_8_20261101/wheel_urls.json').read_text())
urls+=['https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.16/flash_attn-2.8.3+cu128torch2.7-cp311-cp311-linux_x86_64.whl']
for url in urls:
 name=unquote(urlsplit(url).path.rsplit('/',1)[-1]);p=B/name
 if p.exists():continue
 resp=requests.get(url,headers={'Range':'bytes=0-0'},stream=True,timeout=30);resp.raise_for_status()
 size=int(resp.headers['Content-Range'].split('/')[-1]);signed=resp.url;resp.close()
 tmp=p.with_suffix('.part')
 with tmp.open('wb') as f:f.truncate(size)
 def fetch(a):
  end=min(a+4*1024**2,size)-1
  for k in range(6):
   try:
    with requests.get(signed,headers={'Range':f'bytes={a}-{end}'},stream=True,timeout=(15,90)) as r:
     r.raise_for_status();assert r.status_code==206
     assert r.headers.get('Content-Range','').startswith(f'bytes {a}-{end}/')
     with tmp.open('r+b') as f:
      f.seek(a);n=0
      for x in r.iter_content(1024**2):f.write(x);n+=len(x)
     assert n==end-a+1
    return n
   except:
    if k==5:raise
    time.sleep(2)
 with concurrent.futures.ThreadPoolExecutor(24) as pool:
  total=0;t=time.time()
  for n in pool.map(fetch,range(0,size,4*1024**2)):
   total+=n
   if total%(64*1024**2)==0:print(name,total//1024**2,round(total/(time.time()-t)/1024**2,1),flush=True)
 sha=urlsplit(url).fragment.removeprefix('sha256=')
 h=hashlib.sha256()
 with tmp.open('rb') as f:
  while x:=f.read(16*1024**2):h.update(x)
 if sha:assert h.hexdigest()==sha
 tmp.rename(p);print('WHEEL_READY',name,size,h.hexdigest(),flush=True)
