set -euo pipefail
source /data5/shenglin.zeng/env.sh
export CUDA_VISIBLE_DEVICES=''
export TMPDIR=/data5/shenglin.zeng/tmp
export PYTHONPATH=/data5/shenglin.zeng/tools/eval_transfer_20261025
cd /data5/shenglin.zeng/sdxl_crossdistill
nice -n 19 ionice -c 2 -n 7 python - <<'PY'
from pathlib import Path
import conda_pack,json,tarfile,hashlib,shutil,ast
root=Path.cwd();bundle=root/'exports/coco_uniform_g160_20261025';bundle.mkdir(parents=True,exist_ok=True)
env=bundle/'environment.tar.gz'
if not env.exists():
 with tarfile.open(env,'w:gz',compresslevel=1) as tar:
  tar.add('/data5/shenglin.zeng/envs/crossdistill/lib/python3.12/site-packages',arcname='lib/python3.12/site-packages')
print('ENV_PACKED',env.stat().st_size,flush=True)
old=root/'outputs/coco10k_g160_20261020'
source=root/'outputs/joint4_uniform_10k_2gpu_mb16_20261023/code/sdxl_research'
ck=root/'outputs/joint4_uniform_10k_2gpu_mb16_20261023/train/weights/20261024T063356+0800_g000160_periodic'
config=json.loads((ck.parent.parent/'config.json').read_text())
assert json.loads((ck/'complete.json').read_text())['student_updates']==160
# The only intended sampling change versus old students is the checkpoint schedule.
def ast_function(path,name):
 t=ast.parse(path.read_text());n=next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name==name)
 return ast.dump(n,include_attributes=False)
baseline=root/'sdxl_coco10k/evaluate.py'
if baseline.exists():
 for name in ('transformed','features','frechet_distance'):
  assert ast_function(source/'evaluation_core.py',name)==ast_function(baseline,name),name
staged=bundle/'meta';staged.mkdir(exist_ok=True)
shutil.copyfile(old/'manifest.json',staged/'old_manifest.json')
shutil.copyfile(old/'scores.json',staged/'old_scores.json')
shutil.copyfile(old/'complete.json',staged/'old_complete.json')
(staged/'config.json').write_text(json.dumps(config,indent=2))
assets=bundle/'assets.tar'
if not assets.exists():
 with tarfile.open(assets,'w') as tar:
  for name in ('models/sdxl-base-1.0','models/metrics/clean_fid','datasets/dmd2/coco10k.zip'):
   tar.add(root/name,arcname=name)
  tar.add(old/'features/real',arcname='reference_features/real')
  for name in ('evaluate.py','evaluation_core.py','objectives.py'):
   tar.add(source/name,arcname='code/'+name)
  tar.add(ck,arcname='checkpoints/weights/g160')
  tar.add(staged/'config.json',arcname='checkpoints/config.json')
  for name in ('old_manifest.json','old_scores.json','old_complete.json'):
   tar.add(staged/name,arcname='reference/'+name)
 print('ASSETS_PACKED',assets.stat().st_size,flush=True)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
metadata={p.name:{'bytes':p.stat().st_size,'sha256':sha(p)} for p in (env,assets)}
(bundle/'transfer_manifest.json').write_text(json.dumps(metadata,indent=2))
print('TRANSFER_MANIFEST',json.dumps(metadata),flush=True)
PY

