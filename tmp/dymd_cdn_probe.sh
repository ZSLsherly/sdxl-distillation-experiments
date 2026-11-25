export http_proxy=http://127.0.0.1:18897 https_proxy=http://127.0.0.1:18897
/data1/shenglin.zeng/miniconda3/envs/sherly/bin/python - <<'PY'
import time,requests,concurrent.futures
from huggingface_hub import get_hf_file_metadata,hf_hub_url
m=get_hf_file_metadata(hf_hub_url('DAGroup-PKU/PF_Wan','backbone.pth',revision='8fa7e17d5f944d0275df7dc7929137b956367ff8'))
def probe(use_proxy):
 s=requests.Session();s.trust_env=use_proxy;t=time.time()
 try:
  with s.get(m.location,headers={'Range':'bytes=0-1048575'},timeout=(8,20),stream=True) as r:
   r.raise_for_status();n=sum(len(x) for x in r.iter_content(1024**2))
   print('CDN',use_proxy,r.status_code,n,round(time.time()-t,2),flush=True)
 except Exception as e: print('CDN',use_proxy,type(e).__name__,str(e)[:100],flush=True)
with concurrent.futures.ThreadPoolExecutor(2) as p:list(p.map(probe,[True,False]))
PY
