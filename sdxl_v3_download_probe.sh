source /data5/shenglin.zeng/env.sh
cd /data5/shenglin.zeng/sdxl_crossdistill
python - <<'PY'
import requests,time
url='https://huggingface.co/facebook/dinov2-small/resolve/ed25f3a31f01632728cabb09d1542f84ab7b0056/model.safetensors'
s=requests.Session();s.trust_env=False
for host in [url,url.replace('huggingface.co','hf-mirror.com')]:
 try:
  t=time.time();r=s.get(host,stream=True,timeout=(8,12));r.raise_for_status();chunk=next(r.iter_content(1024*1024));print('DIRECT_OK',len(chunk),round(time.time()-t,2),host.split('/')[2],flush=True);r.close();break
 except Exception as e:print('DIRECT_FAIL',type(e).__name__,host.split('/')[2],flush=True)
PY
