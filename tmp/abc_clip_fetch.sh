set -eu
. /root/sdxl_research/env.sh
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$SDXL_WORK/semantic_abc_20261103/model"
cd "$SDXL_WORK/semantic_abc_20261103/model"
python -u - <<'PY'
import json,hashlib,requests
from pathlib import Path
repo='openai/clip-vit-base-patch32'; base='https://hf-mirror.com'
r=requests.get(f'{base}/api/models/{repo}?blobs=true',timeout=35);r.raise_for_status()
info=r.json(); fs={v['rfilename']:v for v in info['siblings']};rev=info['sha']
for name in ('config.json','preprocessor_config.json','tokenizer_config.json','vocab.json','merges.txt','pytorch_model.bin'):
    dest=Path(name); part=dest.with_suffix('.partial'); s=fs[name]
    def valid(p):
        if not p.is_file() or p.stat().st_size!=s['size']:return False
        if name=='pytorch_model.bin':
            with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()=='a63082132ba4f97a80bea76823f544493bffa8082296d62d71581a4feff1576f'
        return hashlib.sha1(f'blob {s["size"]}\0'.encode()+p.read_bytes()).hexdigest()==s['blobId']
    if not valid(dest):
        for origin in (base,'https://huggingface.co',base):
            try:
                with requests.get(f'{origin}/{repo}/resolve/{rev}/{name}?download=true',timeout=(10,70),stream=True) as res:
                    res.raise_for_status()
                    with part.open('wb') as f:
                        for chunk in res.iter_content(1<<20):f.write(chunk)
                assert valid(part)
                part.replace(dest);break
            except Exception as e:print('attempt',name,type(e).__name__,flush=True)
        else:raise RuntimeError(name)
    print('verified',name,flush=True)
Path('revision.txt').write_text(rev)
PY
