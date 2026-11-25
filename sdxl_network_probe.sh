source /data5/shenglin.zeng/env.sh
python - <<'PY'
import requests,concurrent.futures,time
urls=['https://hf-mirror.com/api/models/stabilityai/stable-diffusion-xl-base-1.0','https://huggingface.co/api/models/stabilityai/stable-diffusion-xl-base-1.0','https://images.cocodataset.org/annotations/annotations_trainval2014.zip','http://images.cocodataset.org/annotations/annotations_trainval2014.zip','https://raw.githubusercontent.com/tianweiy/DMD2/main/README.md']
def probe(url):
 t=time.time()
 try:
  r=requests.get(url,stream=True,timeout=(8,12));print(url,r.status_code,dict((k,v) for k,v in r.headers.items() if k.lower() in ['content-length','content-type']),round(time.time()-t,1),flush=True);r.close()
 except Exception as e:print(url,type(e).__name__,str(e)[:100],flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=5) as p:list(p.map(probe,urls))
PY
