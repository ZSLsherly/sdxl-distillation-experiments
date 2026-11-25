set -euo pipefail
source /data5/shenglin.zeng/env.sh
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export PYTHONPATH=/data5/shenglin.zeng/tools/eval_transfer_20261025
python -u - <<'PY'
import sys,json,tarfile,hashlib,os
from pathlib import Path
import torch
root=Path('/data5/shenglin.zeng/sdxl_crossdistill')
out=root/'exports/turning_20261026';out.mkdir(parents=True,exist_ok=True)
prefix=Path(sys.prefix);base=Path(sys.base_prefix)
print('PYTHON_PREFIX',prefix,'BASE',base,'CONDA',(prefix/'conda-meta').exists(),flush=True)
d=torch.load(root/'datasets/prepared.pt',weights_only=True,map_location='cpu')
ids=list(range(32))
small={'prompts':[d['prompts'][i] for i in ids],'source_indices':ids}
for k in ('prompt_embeds','negative_prompt_embeds','pooled_prompt_embeds','negative_pooled_prompt_embeds','time_ids'):
 small[k]=d[k] if len(d[k])==1 else d[k][ids]
torch.save(small,out/'conditions.pt')
(out/'prompts.json').write_text(json.dumps({'source':'datasets/prepared.pt','indices':ids,'prompts':small['prompts'],'seeds':[201,202]},ensure_ascii=False,indent=2))
model=root/'models/sdxl-base-1.0'
assets=out/'sdxl_sampling.tar'
if not assets.exists():
 with tarfile.open(assets,'w') as tar:
  for folder in ('unet','vae','scheduler'):
   files=[f for f in (model/folder).iterdir() if f.is_file() and (f.suffix=='.json' or '.fp16.' in f.name)]
   if folder!='scheduler':assert any('.fp16.' in f.name for f in files),folder
   for f in files:tar.add(f,arcname='models/sdxl-base-1.0/'+folder+'/'+f.name)
  tar.add(out/'conditions.pt',arcname='conditions.pt')
  tar.add(out/'prompts.json',arcname='prompts.json')
print('ASSETS_READY',assets.stat().st_size,flush=True)
env=out/'environment.tar.gz'
if not env.exists():
 if (prefix/'conda-meta').exists():
  import conda_pack
  conda_pack.pack(prefix=str(prefix),output=str(env),compress_level=1)
 else:
  # This project's venv already has its own portable Python runtime.
  with tarfile.open(env,'w:gz',compresslevel=1) as tar:
   tar.add(base,arcname='runtime')
   tar.add(prefix/'lib/python3.12/site-packages',arcname='env/lib/python3.12/site-packages')
print('ENV_READY',env.stat().st_size,flush=True)
manifest={}
for p in (assets,env):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
 manifest[p.name]={'bytes':p.stat().st_size,'sha256':h.hexdigest()}
(out/'manifest.json').write_text(json.dumps(manifest,indent=2))
print('EXPORT_READY',json.dumps(manifest),flush=True)
PY
