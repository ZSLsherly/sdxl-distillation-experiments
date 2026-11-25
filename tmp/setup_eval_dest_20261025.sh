set -euo pipefail
cd /data1/shenglin.zeng/sdxl_eval_uniform_20261025
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME="$PWD/cache"
export LD_LIBRARY_PATH="$PWD/runtime/lib:${LD_LIBRARY_PATH:-}"
runtime/bin/python - <<'PY'
import hashlib,json,tarfile
from pathlib import Path
root=Path.cwd();manifest=json.loads((root/'transfers/transfer_manifest.json').read_text())
for name,m in manifest.items():
 p=root/'transfers'/name;h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 assert p.stat().st_size==m['bytes'] and h.hexdigest()==m['sha256'],name
 print('VERIFIED',name,flush=True)
PY
runtime/bin/python -m venv --copies --without-pip env
nice -n 19 ionice -c 2 -n 7 tar -xzf transfers/environment.tar.gz -C env
nice -n 19 ionice -c 2 -n 7 tar -xf transfers/assets.tar
env/bin/python - <<'PY'
import json,sys,importlib.metadata as md
from pathlib import Path
import torch,torchvision,diffusers,transformers,numpy,scipy,cleanfid
versions={k:md.version(k) for k in ['torch','torchvision','diffusers','transformers','accelerate','numpy','scipy','safetensors','Pillow','clean-fid']}
versions['python']=sys.version
Path('environment_versions.json').write_text(json.dumps(versions,indent=2));print('VERSIONS',versions,flush=True)
assert torch.__version__=='2.5.1+cu121'
assert not torch.cuda.is_available()
Path('models.json').write_text(json.dumps({'joint4_uniform':{'checkpoint':'checkpoints/weights/g160'}}))
PY
nice -n 19 env/bin/python code/evaluate.py prepare --root "$PWD" --out "$PWD/outputs/coco10k" --models "$PWD/models.json" --n 10000
env/bin/python - <<'PY'
import json
from pathlib import Path
import numpy as np
root=Path.cwd();out=root/'outputs/coco10k'
old=json.loads((root/'reference/old_manifest.json').read_text());new=json.loads((out/'manifest.json').read_text())
for k in ['n','source_zip_sha256','csv_sha256']:assert old[k]==new[k],k
fields=('index','prompt','real_file','seed')
assert [tuple(r[k] for k in fields) for r in old['samples']]==[tuple(r[k] for k in fields) for r in new['samples']]
new['visual_indices']=old['visual_indices']
(out/'manifest.json').write_text(json.dumps(new,ensure_ascii=False,indent=2))
ids=[]
for p in sorted((root/'reference_features/real').glob('*.npz')):
 with np.load(p) as z:
  ids.extend(z['indices'].tolist())
  for k in ('full','patch'):assert z[k].shape==(len(z['indices']),2048) and np.isfinite(z[k]).all()
assert ids==list(range(10000))
(out/'features').mkdir(exist_ok=True)
(out/'features/real').symlink_to(root/'reference_features/real',target_is_directory=True)
verified={'n':10000,'same_prompt_seed_real_order':True,'same_source_zip':True,'same_csv':True,'real_feature_indices_complete':True,'schedule':new['models']['joint4_uniform']['config']['schedule']}
(out/'protocol_verification.json').write_text(json.dumps(verified,indent=2));print('PROTOCOL_VERIFIED',verified,flush=True)
PY
