R=/data1/shenglin.zeng/dmd2_matched_10k_20261119
CUDA_VISIBLE_DEVICES="" "$R/env/bin/python" -u - <<'PY'
import requests
rev='be22767697a1f3ca656b73c776e15fa335c86c6c'
folder='data/laion_vae_latents/sdxl_vae_latents_laion_500k'
for host in ['https://huggingface.co','https://hf-mirror.com']:
 try:
  r=requests.get(f'{host}/api/models/tianweiy/DMD2/tree/{rev}/{folder}?recursive=false&expand=false',timeout=(8,25));r.raise_for_status()
  rows=[x for x in r.json() if x['path'].endswith(tuple(f'{i:03d}.pt' for i in range(2,9)))]
  print('SOURCE_METADATA',host,rows,flush=True)
  break
 except Exception as e: print(type(e).__name__,str(e)[:160],flush=True)
else:raise RuntimeError('No dataset metadata endpoint reachable')
PY
