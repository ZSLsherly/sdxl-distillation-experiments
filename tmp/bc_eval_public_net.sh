set -eu
. /root/sdxl_research/env.sh
python - <<'PY'
import requests,concurrent.futures
urls=['https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/text_encoder/config.json','https://hf-mirror.com/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/text_encoder/config.json','https://huggingface.co/api/models/stabilityai/stable-diffusion-xl-base-1.0?blobs=true']
def check(url):
 try:
  r=requests.get(url,timeout=5);return [url,r.status_code,r.text[:60]]
 except Exception as e:return [url,str(e)[:200]]
print(list(concurrent.futures.ThreadPoolExecutor(3).map(check,urls)))
PY
