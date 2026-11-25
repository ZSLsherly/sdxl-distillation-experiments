set -e
source /data5/shenglin.zeng/env.sh
cd /data5/shenglin.zeng/sdxl_crossdistill
python - <<'PY'
from pathlib import Path
import requests,hashlib,time
p=Path('models/dinov2-small');s=requests.Session();s.trust_env=False
urls=[(p/'download_url.txt').read_text().strip(),'https://hf-mirror.com/facebook/dinov2-small/resolve/ed25f3a31f01632728cabb09d1542f84ab7b0056/model.safetensors']
for url in urls:
 try:
  start=time.time();h=hashlib.sha256();total=0
  with s.get(url,stream=True,timeout=(10,60)) as r:
   r.raise_for_status()
   with (p/'model.safetensors.direct').open('wb') as f:
    for b in r.iter_content(1024**2):
     f.write(b);h.update(b);total+=len(b)
     if total%(10*1024**2)==0:print('DOWNLOADED_MB',total//1024**2,round(time.time()-start),flush=True)
  assert total==88249960 and h.hexdigest()=='ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1'
  (p/'model.safetensors.direct').replace(p/'model.safetensors');print('DINO_VERIFIED',total,flush=True);break
 except Exception as e:print('DOWNLOAD_FAILED',type(e).__name__,flush=True)
else:raise RuntimeError('Both direct downloads failed')
PY
