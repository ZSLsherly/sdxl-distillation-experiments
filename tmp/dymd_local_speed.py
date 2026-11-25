import requests,time,concurrent.futures
url='https://huggingface.co/DAGroup-PKU/PF_Wan/resolve/8fa7e17d5f944d0275df7dc7929137b956367ff8/backbone.pth'
r=requests.head(url,allow_redirects=True,timeout=30);r.raise_for_status();signed=r.url
def run(i):
 t=time.time();a=i*1024**2
 with requests.get(signed,headers={'Range':f'bytes={a}-{a+1024**2-1}'},stream=True,timeout=(10,30)) as q:
  q.raise_for_status();n=0
  for x in q.iter_content(1024**2):
   n+=len(x)
   if n>=1024**2:break
 print(i,q.status_code,n,round(time.time()-t,2),flush=True)
with concurrent.futures.ThreadPoolExecutor(4) as p:list(p.map(run,range(4)))
