python3 -u - <<'PY'
import urllib.request,urllib.error
url='https://hf-mirror.com/tianweiy/DMD2/resolve/be22767697a1f3ca656b73c776e15fa335c86c6c/data/laion_vae_latents/sdxl_vae_latents_laion_500k/vae_latents_004.pt?download=true'
try:
 r=urllib.request.build_opener(urllib.request.ProxyHandler({})).open(urllib.request.Request(url,headers={'Range':'bytes=805306368-822083583'}),timeout=15)
 print('HTTP',r.status,'RANGE',r.headers.get('Content-Range'),'BYTE',len(r.read(1)));r.close()
except urllib.error.HTTPError as e:print('HTTP_ERROR',e.code,e.reason,'RETRY_AFTER',e.headers.get('Retry-After'),e.read(200).decode(errors='replace'))
except Exception as e: print(type(e).__name__,str(e)[:150])
PY
