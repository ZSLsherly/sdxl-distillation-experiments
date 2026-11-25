set -euo pipefail
cd /data1/shenglin.zeng/sdxl_eval_uniform_20261025
export CUDA_VISIBLE_DEVICES=''
export LD_LIBRARY_PATH="$PWD/runtime/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
nice -n 19 env/bin/python - <<'PY'
from pathlib import Path
import hashlib,json,tarfile
root=Path.cwd();p=root/'joint34_transfers/training_assets.tar'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
m=json.loads((root/'joint34_transfers/transfer_manifest.json').read_text())[p.name]
assert p.stat().st_size==m['bytes'] and sha(p)==m['sha256']
with tarfile.open(p) as t:t.extractall(root,filter='data')
proof=json.loads((root/'joint34_reference_provenance.json').read_text())
assert sha(root/'datasets/parallel_10k_seed20261018/prepared.pt')==proof['prepared_pt_sha256']
for name,h in proof['source_sha256'].items():assert sha(root/'training_code'/name)==h,name
print('DATA_AND_FROZEN_SOURCE_VERIFIED',flush=True)
PY
env/bin/python - <<'PY'
import ast,json,sys,py_compile,os
from pathlib import Path
import torch
root=Path.cwd();source=root/'training_code';sys.path.insert(0,str(source))
from launch import flags
from stage_policy import make_stage_plan
import train
proof=json.loads((root/'joint34_reference_provenance.json').read_text())
allowed={n.args[0].value[2:].replace('-','_') for n in ast.walk(ast.parse((source/'train.py').read_text())) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='add_argument' and n.args and isinstance(n.args[0],ast.Constant) and isinstance(n.args[0].value,str) and n.args[0].value.startswith('--')}
out=root/'outputs/joint34_uniform_10k_2gpu_20261025';out.mkdir(parents=True,exist_ok=False)
for phase,key,max_g in [('phase1','initial_config',2),('train','reference_config',160)]:
 ref=proof[key];cfg={k:v for k,v in ref.items() if k in allowed and k not in ('root','run','resume','resume_repack')}
 cfg.update(variant='joint34',gan_stages=[3,4],schedule=ref['schedule'][:-1],data_dir='datasets/parallel_10k_seed20261018',max_student_updates=max_g)
 sys.argv=['train.py','--root',str(root),'--run','joint34_uniform_10k_2gpu_20261025/'+phase,*flags(cfg)]
 args=train.arguments();assert args.batch==64 and args.max_student_updates==max_g and args.gan_stages==[3,4]
 different={k:{'reference':ref.get(k),'new':v} for k,v in cfg.items() if v!=ref.get(k)}
 assert set(different)<= {'variant','gan_stages','schedule','data_dir','max_student_updates'},different
 (out/(phase+'_config.json')).write_text(json.dumps(cfg,indent=2));print('CONFIG_DIFF',phase,json.dumps(different),flush=True)
plan=make_stage_plan('joint34');assert sum(plan.student_stage(i) in plan.gan for i in range(160))==106
data=torch.load(root/'datasets/parallel_10k_seed20261018/prepared.pt',map_location='cpu',weights_only=True,mmap=True)
assert data['train_count']==10000
for p in source.glob('*.py'):py_compile.compile(str(p),doraise=True)
mem={line.split(':')[0]:int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')}
assert mem['MemAvailable']>100*1024*1024,mem
design={'reference':'joint4_uniform_10k_2gpu_mb16_20261023','initialization':'fresh SDXL Base for phase1; own G2 full-state resume for G3-G160','changed_training_objective':{'gan_stages':[[4],[3,4]]},'same_global_batch':64,'world_size':2,'microbatch_schedule':{'G1_G2':4,'G3_G160':16},'GAN_student_updates':106,'fake_D_updates':800,'D_fake_stage':4,'checkpoint_updates':[40,80,120,160],'final_full_training_state':True,'training_data_sha256':proof['prepared_pt_sha256'],'hardware_note':'reference: two Ampere A100-labelled 40GB GPUs; new: two RTX4090-labelled 48GB GPUs; same core dependencies, no bitwise identity claim','python':sys.version,'torch':torch.__version__}
(out/'design.json').write_text(json.dumps(design,indent=2));print('CPU_PREFLIGHT_PASSED',json.dumps(design),flush=True)
PY
