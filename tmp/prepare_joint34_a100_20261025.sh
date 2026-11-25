set -euo pipefail
source /data5/shenglin.zeng/env.sh
export CUDA_VISIBLE_DEVICES=''
cd /data5/shenglin.zeng/sdxl_crossdistill
python - <<'PY'
from pathlib import Path
import ast,json,sys,hashlib
r=Path.cwd();source=r/'outputs/joint4_uniform_10k_2gpu_mb16_20261023/code/sdxl_research'
out=r/'outputs/joint34_uniform_10k_2gpu_20261025';out.mkdir(parents=True,exist_ok=False)
proof=json.loads((r/'exports/joint34_uniform_20261025/provenance.json').read_text())
for n,h in proof['source_sha256'].items():assert hashlib.sha256((source/n).read_bytes()).hexdigest()==h,n
sys.path.insert(0,str(source));from launch import flags
import train
allowed={n.args[0].value[2:].replace('-','_') for n in ast.walk(ast.parse((source/'train.py').read_text())) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='add_argument' and n.args and isinstance(n.args[0],ast.Constant) and isinstance(n.args[0].value,str) and n.args[0].value.startswith('--')}
for phase,key,max_g in [('phase1','initial_config',2),('train','reference_config',160)]:
 ref=proof[key];cfg={k:v for k,v in ref.items() if k in allowed and k not in ('root','run','resume','resume_repack')}
 cfg.update(variant='joint34',gan_stages=[3,4],schedule=ref['schedule'][:-1],data_dir='datasets/parallel_10k_seed20261018',max_student_updates=max_g)
 sys.argv=['train.py','--root',str(r),'--run',out.name+'/'+phase,*flags(cfg)];a=train.arguments()
 assert a.batch==64 and a.gan_stages==[3,4] and a.fake_offload and a.offload_student_optimizer
 (out/(phase+'_config.json')).write_text(json.dumps(cfg,indent=2))
design={'reference':'joint4_uniform_10k_2gpu_mb16_20261023','initialization':'fresh SDXL Base; own G2 state then G3-G160','changed_training_objective':{'gan_stages':[[4],[3,4]]},'world_size':2,'same_global_batch':64,'microbatch_schedule':{'G1_G2':4,'G3_G160':16},'fake_offload':True,'offload_student_optimizer':True,'GAN_student_updates':106,'fake_D_updates':800,'checkpoint_updates':[40,80,120,160],'final_full_training_state':True,'training_data_sha256':proof['prepared_pt_sha256'],'server':'172.18.33.60','queue_policy':'Wait up to 24 hours for two GPUs idle for at least 60 seconds and host MemAvailable >=120 GiB. Do not reserve one GPU alone. Recheck immediately before launch. Stop own processes on collision.','runtime_note':'Same original server, native environment and two-GPU runtime as reference. Previous 4090 attempt stopped before G1 due to shared user.slice memory pressure; four-GPU preparation cancelled before launch.'}
(out/'design.json').write_text(json.dumps(design,indent=2));print('A100_MATCHED_CONFIG_PREPARED',json.dumps(design))
PY
