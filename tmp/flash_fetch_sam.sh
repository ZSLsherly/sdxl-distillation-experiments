set -eu
. /root/sdxl_research/env.sh
d="$SDXL_WORK/flash_gan_inputs/sam-vit-base"
mkdir -p "$d"
cd "$d"
python - <<'PY'
import hashlib,json,pathlib,requests
rev='70c1a07f894ebb5b307fd9eaaee97b9dfc16068f'
for name in ('config.json','model.safetensors'):
    dst=pathlib.Path(name);part=dst.with_suffix('.part')
    for base in ('https://hf-mirror.com','https://huggingface.co'):
        try:
            with requests.get(f'{base}/facebook/sam-vit-base/resolve/{rev}/{name}?download=true',timeout=(10,80),stream=True) as r:
                r.raise_for_status()
                with part.open('wb') as f:
                    for chunk in r.iter_content(1024**2):f.write(chunk)
            if name.endswith('json'):assert json.loads(part.read_text())['model_type']=='sam'
            else:
                with part.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()=='892c410e496344e527255ccdcb2cb7244a609acb5389c7c4fdba1288f861c579'
            part.replace(dst);print('verified',name,dst.stat().st_size,flush=True);break
        except Exception as e:print(type(e).__name__,flush=True)
    else:raise SystemExit('SAM download failed')
PY
