set -euo pipefail
root=/data1/shenglin.zeng/teacher_turning_20261026
cd "$root"
export CUDA_VISIBLE_DEVICES=''
export TMPDIR="$root/tmp"
python3 - <<'PY'
import json,hashlib,os,tarfile
from pathlib import Path
root=Path.cwd();manifest=json.loads((root/'transfers/manifest.json').read_text())
for name,m in manifest.items():
 p=root/'transfers'/name;h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 assert h.hexdigest()==m['sha256'] and p.stat().st_size==m['bytes'],name
 print('VERIFIED',name,flush=True)
 with tarfile.open(str(p)) as tar:
  for member in tar.getmembers():
   assert not member.name.startswith('/') and '..' not in Path(member.name).parts
  tar.extractall(str(root))
PY
export LD_LIBRARY_PATH="$root/runtime/lib:${LD_LIBRARY_PATH:-}"
runtime/bin/python3.12 -m venv --copies --without-pip env
env/bin/python -u - <<'PY'
import torch,diffusers,transformers,json
from pathlib import Path
d=torch.load('conditions.pt',weights_only=True,map_location='cpu')
assert len(d['prompts'])==32
print('RUNTIME',torch.__version__,diffusers.__version__,transformers.__version__,flush=True)
print('PROMPTS',len(d['prompts']),d['source_indices'],flush=True)
for folder in ('unet','vae'):
 files=list(Path('models/sdxl-base-1.0',folder).glob('*.fp16.safetensors'))
 assert files and all(p.exists() and p.stat().st_size>1000 for p in files)
Path('setup_complete.json').write_text(json.dumps({'torch':torch.__version__,'prompts':32,'ready':True}))
PY
