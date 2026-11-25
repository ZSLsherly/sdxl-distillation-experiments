import sys,json
from pathlib import Path
sys.path.insert(0,'.image_runtime_pkgs')
import requests
r=requests.get('https://api.github.com/repos/tianweiy/DMD2/git/trees/main?recursive=1',timeout=45);r.raise_for_status()
paths=[x['path'] for x in r.json()['tree'] if x['path'].endswith('.py') and any(s in x['path'] for s in ['dataset','vae_latent','sdxl_guidance'])]
print('PATHS',paths)
dest=Path('tmp/sdxl_reference');dest.mkdir(parents=True,exist_ok=True)
for p in paths:
 if 'sdxl' in p or 'text_dataset' in p:
  r=requests.get('https://raw.githubusercontent.com/tianweiy/DMD2/main/'+p,timeout=45);r.raise_for_status()
  (dest/Path(p).name).write_text(r.text,encoding='utf-8')
  print('SAVED',p)
