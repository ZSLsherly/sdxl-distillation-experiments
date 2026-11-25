import json
from pathlib import Path
root=Path(__file__).resolve().parent.parent/'sdxl_research/configs'
root.mkdir(exist_ok=True)
base=dict(variant='joint4',data_dir='datasets/parallel_10k_seed20261018',batch=64,
 micro_batch=16,rollout_batch=16,vae_batch=4,max_student_updates=160,
 save_every_updates=40,eval_every_samples=128,hours=24,student_lr=1e-6,fake_lr=1e-6,
 d_lr=1e-4,fake_updates=5,gan_weight=0.005274229682981968,teacher_cfg=8,
 seed=20261017,pcm_weight=10,schedule=[999,749,499,249],dmd_t_max=919,
 fake_offload=True,shard_granularity='fine',d_grad_bucket_mb=0,control_mode='legacy')
def write(path,value):
 path.parent.mkdir(parents=True,exist_ok=True)
 path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
for name,overrides in {
 'uniform_joint4':{},'uniform_dmd':{'variant':'dmd_only'},
 'legacy_joint4':{'schedule':[999,919,499,199]},
 'legacy_dmd':{'schedule':[999,919,499,199],'variant':'dmd_only'},
 'uniform_gan4':{'variant':'gan4'},'uniform_joint34':{'variant':'joint34'},
 'uniform_joint4_gan_half':{'gan_weight':base['gan_weight']/2},
 'uniform_joint4_gan_double':{'gan_weight':base['gan_weight']*2},
 'sampler_only_joint4':{'pcm_boundary':919,'dmd_t_max':919},
 'uniform_aligned_dmd749':{'dmd_t_max':749},
 'uniform_joint4_2gpu_memory':{'micro_batch':4,'rollout_batch':4,'offload_student_optimizer':True},
}.items():write(root/f'{name}.json',base|overrides)
perf=base|dict(benchmark=True,benchmark_batch=64,benchmark_cycles=9,benchmark_warmup=3)
for name,overrides in {
 'legacy':{'d_grad_bucket_mb':0,'control_mode':'legacy'},
 'bucket_cycle':{'d_grad_bucket_mb':16,'control_mode':'cycle'},
 'block':{'shard_granularity':'block','d_grad_bucket_mb':16,'control_mode':'cycle'},
 'foreach':{'fake_adam_foreach':True},
 'teacher_cfg':{'batched_teacher_cfg':True},
 'compile_teacher':{'compile_teacher':True},
}.items():write(root/'performance'/f'{name}.json',perf|overrides)
write(root/'performance'/'trace.json',perf|{'trace_cycles':2,'benchmark_cycles':4,'benchmark_warmup':3})
write(root/'models.example.json',{'uniform_joint4':{'checkpoint':'outputs/research/REPLACE_RUN/weights/REPLACE_G160_CHECKPOINT'},'sdxl':{},'dmd2':{}})
print('wrote',len(list(root.rglob('*.json'))),'configs')
