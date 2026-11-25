import json,socket,time,concurrent.futures
from pathlib import Path
import requests
d=json.loads(Path('/tmp/dymd_network_probe.json').read_text())
orig=socket.getaddrinfo
def resolve(host,port,*a,**k):
 if host in d['mapping'] and d['mapping'][host]:host=d['mapping'][host][0]
 return orig(host,port,*a,**k)
socket.getaddrinfo=resolve
def probe(url):
 s=requests.Session();s.trust_env=False;t=time.time()
 try:
  with s.get(url,headers={'Range':'bytes=0-1048575'},stream=True,timeout=(8,20)) as r:
   r.raise_for_status();n=0
   for x in r.iter_content(1024**2):
    n+=len(x)
    if n>=1024**2:break
   print(r.status_code,n,round(time.time()-t,2),flush=True)
 except Exception as e: print(type(e).__name__,str(e).split('url:')[0][:120],flush=True)
with concurrent.futures.ThreadPoolExecutor(2) as p:list(p.map(probe,[d['url'],'https://huggingface.co/api/models/DAGroup-PKU/PF_Wan']))
