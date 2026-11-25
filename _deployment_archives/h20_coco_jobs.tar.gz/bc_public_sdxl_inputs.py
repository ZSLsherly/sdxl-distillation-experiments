"""Fetch only published SDXL text-encoding inputs into the data disk."""
import hashlib,json,time
from pathlib import Path
import requests

root=Path('/root/autodl-tmp/sdxl_runs/bc_texture_inputs')
base=root/'models/sdxl-base-1.0'
base.mkdir(parents=True,exist_ok=True)
for name in ('unet','vae','scheduler'):
    link=base/name
    if not link.exists():link.symlink_to(Path('/root/sdxl_research/models/sdxl-base-1.0')/name,target_is_directory=True)
repo='stabilityai/stable-diffusion-xl-base-1.0'
endpoint='https://hf-mirror.com'
meta=requests.get(f'{endpoint}/api/models/{repo}?blobs=true',timeout=30);meta.raise_for_status()
info=meta.json(); revision=info['sha'];files={x['rfilename']:x for x in info['siblings']}
names=['model_index.json']
for enc in ('text_encoder','text_encoder_2'):
    names.extend([enc+'/config.json',enc+'/model.fp16.safetensors'])
for tok in ('tokenizer','tokenizer_2'):
    names.extend(tok+'/'+n for n in ['vocab.json','merges.txt','tokenizer_config.json','special_tokens_map.json'])
for name in names:
    spec=files[name]; dest=base/name;dest.parent.mkdir(exist_ok=True)
    size=spec['size'];lfs=spec.get('lfs',{})
    def valid(p):
        if not p.is_file() or p.stat().st_size!=size:return False
        if lfs:
            with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()==lfs['sha256']
        return hashlib.sha1(f'blob {size}\0'.encode()+p.read_bytes()).hexdigest()==spec['blobId']
    if valid(dest):continue
    part=dest.with_suffix(dest.suffix+'.partial')
    for attempt in range(3):
        try:
            with requests.get(f'{endpoint}/{repo}/resolve/{revision}/{name}',stream=True,timeout=(10,90)) as response:
                response.raise_for_status()
                with part.open('wb') as f:
                    for block in response.iter_content(1<<20):f.write(block)
            if not valid(part):raise ValueError(f'Published checksum mismatch: {name}')
            part.replace(dest);print('VERIFIED',name,size,flush=True);break
        except (requests.RequestException,ValueError) as exc:
            if attempt==2:raise
            print('RETRY',name,type(exc).__name__,flush=True);time.sleep(5)
(root/'public_inputs_ready.json').write_text(json.dumps({'repo':repo,'revision':revision,'files':names}))
