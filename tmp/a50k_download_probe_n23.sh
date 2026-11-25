python3 -u - <<'PY'
import urllib.request
for base in ['https://huggingface.co','https://hf-mirror.com']:
 try:
  url=base+'/tianweiy/DMD2/resolve/be22767697a1f3ca656b73c776e15fa335c86c6c/data/laion_vae_latents/sdxl_vae_latents_laion_500k/vae_latents_004.pt?download=true'
  r=urllib.request.urlopen(urllib.request.Request(url,headers={'Range':'bytes=0-1023'}),timeout=12)
  print(base,r.status,r.headers.get('Content-Range'),len(r.read(1024)),flush=True);r.close()
 except Exception as e: print(base,type(e).__name__,str(e)[:160],flush=True)
PY
free -h
find /data3/shenglin.zeng /data1/shenglin.zeng -maxdepth 4 \( -path '*/env/bin/python' -o -name rclone \) -print 2>/dev/null | head -15
