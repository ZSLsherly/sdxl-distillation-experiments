source /data5/shenglin.zeng/env.sh
export HTTPS_PROXY=http://127.0.0.1:18797 HTTP_PROXY=http://127.0.0.1:18797
python - <<'PY'
import requests,time,concurrent.futures
from urllib.parse import urlsplit
from huggingface_hub import get_hf_file_metadata,hf_hub_url
url=hf_hub_url('stabilityai/stable-diffusion-xl-base-1.0','unet/diffusion_pytorch_model.fp16.safetensors',revision='462165984030d82259a11f4367a4eed129e94a7b',endpoint='https://huggingface.co')
meta=get_hf_file_metadata(url,timeout=30)
def probe(name,url):
 session=requests.Session();session.trust_env=False;t=time.time()
 try:
  with session.get(url,headers={'Range':'bytes=0-1048575'},stream=True,timeout=(7,15)) as r:
   r.raise_for_status();n=0
   for b in r.iter_content(1024**2):
    n+=len(b)
    if n>=1024**2:break
   print(name,urlsplit(url).hostname,r.status_code,n,round(time.time()-t,2),flush=True)
 except Exception as e:print(name,type(e).__name__,flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as p:list(p.map(lambda x:probe(*x),[('CDN_DIRECT',meta.location),('MIRROR_DIRECT','https://hf-mirror.com/api/models/stabilityai/stable-diffusion-xl-base-1.0')]))
PY
