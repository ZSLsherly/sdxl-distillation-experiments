python3 -u - <<'PY'
import urllib.request
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
for base in ['https://huggingface.co','https://hf-mirror.com']:
 try:
  url=base+'/tianweiy/DMD2/resolve/be22767697a1f3ca656b73c776e15fa335c86c6c/data/laion_vae_latents/sdxl_vae_latents_laion_500k/vae_latents_004.pt?download=true'
  with opener.open(urllib.request.Request(url,headers={'Range':'bytes=0-1023'}),timeout=8) as r:
   print('DIRECT',base,r.status,r.headers.get('Content-Range'),len(r.read(1024)),flush=True)
 except Exception as e: print('DIRECT',base,type(e).__name__,str(e)[:120],flush=True)
PY
