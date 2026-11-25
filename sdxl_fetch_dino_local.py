import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,'.image_runtime_pkgs')
import requests
root=Path('output/sdxl-crossdistill/dinov2-small');root.mkdir(exist_ok=True)
s=requests.Session()
meta=s.get('https://huggingface.co/api/models/facebook/dinov2-small',timeout=45);meta.raise_for_status();revision=meta.json()['sha']
for name in ['config.json','model.safetensors']:
    p=root/name
    if p.exists():continue
    with s.get(f'https://huggingface.co/facebook/dinov2-small/resolve/{revision}/{name}',stream=True,timeout=(30,120)) as r:
        r.raise_for_status()
        with p.with_suffix('.partial').open('wb') as f:
            for b in r.iter_content(1024**2):f.write(b)
    p.with_suffix('.partial').replace(p);print('DOWNLOADED',name,p.stat().st_size,flush=True)
(root/'source.json').write_text(json.dumps({'repo':'facebook/dinov2-small','revision':revision,'sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [root/'config.json',root/'model.safetensors']}},indent=2))
