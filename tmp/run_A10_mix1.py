from pathlib import Path
import os,sys,fcntl,shutil,json,hashlib
import torch
from diffusers import UNet2DConditionModel
from train import arguments
from launch import command,flags
from run_quota_bc import verify
from queue_dm_pair_coco import read,save,reference,evaluate_checkpoint,timing
import run_dm1_uniform_control as guard
root=Path(os.environ['SDXL_ROOT']);runs=root/'outputs'
job=runs/'A_pcm10_mix1_g160';base=runs/'A_pcm10_g160';prior=runs/'A_mid_clean_x0';code=Path(__file__).resolve().parent
guard.out=job
with (job/'queue.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 cfg=dict(read(base/'effective_config.json'),fake_student_mix=.01)
 sys.argv=['train','--run',job.name+'/model',*flags(cfg)]
 a=arguments()
 assert a.pcm_weight==10 and a.fake_student_mix==.01 and a.fake_updates==2
 assert a.max_student_updates==160 and a.batch==64 and a.turning_profile is None and a.variant=='dmd_only'
 save(job/'effective_config.json',cfg)
 print('CONFIG_VERIFIED; WAITING_FOR A_mid_clean_x0/comparison.json',flush=True)
 with (prior/'queue.lock').open() as gate:
  fcntl.flock(gate,fcntl.LOCK_SH);assert 'clean' in read(prior/'comparison.json')
 model=job/'model'
 if not (model/'status.json').exists():
  if shutil.disk_usage(job).free<7*1024**3:raise RuntimeError('Need 7GiB free')
  if (job/'train.log').exists():raise RuntimeError('Inspect existing attempt')
  guard.run_gpu(command(cfg,root,job.name+'/model',1),job/'train.log')
 ck=Path(verify(model))
 rs=[json.loads(l) for l in (model/'rank0.jsonl').read_text().splitlines()]
 assert [r['g'] for r in rs if r['phase']=='fake_student_mix']==list(range(1,161))
 assert all(r['rate']==.01 for r in rs if r['phase']=='fake_student_mix')
 assert read(model/'data_manifest.json')['train_original_indices']==read(base/'model/data_manifest.json')['train_original_indices']
 state=torch.load(ck/'student_rank0.pt',map_location='cpu',weights_only=True,mmap=True)
 with torch.device('meta'):net=UNet2DConditionModel.from_config(read(root/'models/sdxl-base-1.0/unet/config.json'))
 net.load_state_dict(state,strict=True,assign=True)
 assert all(torch.isfinite(v).all() for v in state.values())
 del state,net
 save(ck/'cpu_validation.json',dict(strict_load=True,finite=True))
 if (job/'CANCEL_POST_TRAIN').exists():
  print('POST_TRAIN_CANCELLED_BY_USER; final weight retained',flush=True)
  sys.exit(0)
 r=evaluate_checkpoint('A_pcm10_mix1',ck,read(model/'config.json'),read(reference/'manifest.json'),job,run_worker=guard.run_gpu)
 baseline=read(base/'comparison.json')['A_pcm10']
 save(job/'comparison.json',dict(A_pcm10=baseline,A_pcm10_mix1=dict(**r,**timing(model)),delta={k:v-baseline[k] for k,v in r.items()}))
 print('COMPLETE',flush=True)

