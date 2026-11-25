"""Wait for the current training, then run the sole newly authorized G160 arm."""
from pathlib import Path
import os,json,time,fcntl,subprocess,math,shutil
from launch import command,gpu_check
from turning_batch64_workflow import descendants,foreign_on,stop_own
root=Path(os.environ['SDXL_ROOT']);code=Path(__file__).resolve().parent
job=root/'outputs/C_mix1_pcm_dm4_fake4_low30_g160';upstream=root/'outputs/A_pcm10_mix1_g160/model'
def read(p):return json.loads(p.read_text())
def main():
 with (job/'queue.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  assert read(job/'cpu_validation.json')['passed']
  cfg=read(job/'effective_config.json')
  print('WAITING_FOR_CURRENT_TRAINING',upstream,flush=True)
  while True:
   if (job/'STOP_REQUEST').exists():return
   s=read(upstream/'status.json')
   if s['status']=='complete' and s['student_updates']==160:break
   time.sleep(30)
  while True:
   if (job/'STOP_REQUEST').exists():return
   try: cards=gpu_check([0],76);break
   except (RuntimeError,subprocess.TimeoutExpired):time.sleep(30)
  while shutil.disk_usage(job).free<7*1024**3:
   if (job/'STOP_REQUEST').exists():return
   print('WAITING_FOR_SPACE',round(shutil.disk_usage(job).free/1024**3,2),flush=True)
   time.sleep(60)
  # Disk waiting may outlive the prior GPU check; check again before launching.
  while True:
   if (job/'STOP_REQUEST').exists():return
   try: cards=gpu_check([0],76);break
   except (RuntimeError,subprocess.TimeoutExpired):time.sleep(30)
  assert not (job/'model').exists(),'Inspect previous run before restarting'
  with (job/'train.log').open('a') as log:
   proc=subprocess.Popen(command(cfg,root,job.name+'/model',1),cwd=code,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   print('TRAIN_STARTED',proc.pid,flush=True)
   try:
    while proc.poll() is None:
     try:conflicts=foreign_on({cards[0][1]},descendants(proc.pid))
     except subprocess.TimeoutExpired:time.sleep(10);continue
     if conflicts:raise RuntimeError('Resource conflict')
     time.sleep(10)
    if proc.returncode:raise RuntimeError(f'Training exit {proc.returncode}')
   except BaseException:stop_own(proc);raise
  model=job/'model';s=read(model/'status.json')
  assert (s['status'],s['student_updates'],s['fake_updates'])==('complete',160,320)
  rows=[json.loads(l) for l in (model/'rank0.jsonl').read_text().splitlines()]
  gs=[r for r in rows if r['phase']=='student_done'];fs=[r for r in rows if r['phase']=='fake']
  assert len(gs)==160 and len(fs)==320
  assert all(g['supervised_stages']==[4] and g['middle_dm_loss']==0 and not g['gan'] for g in gs)
  assert [f['fake_stage'] for f in fs]==[k for g in gs for k in (g['stage'],4)]
  assert all(math.isfinite(f['fake_loss']) and math.isfinite(f['fake_grad_norm']) for f in fs)
  assert all(all(math.isfinite(g[k]) for k in ('pcm_loss','terminal_dm_loss','grad_norm')) for g in gs)
  mixes=[r for r in rows if r['phase']=='fake_student_mix']
  assert [r['g'] for r in mixes]==list(range(1,161)) and all(r['rate']==.01 for r in mixes)
  inp=[r for r in rows if r['phase']=='fake_boundary_input'];assert len(inp)==320
  assert all(r['boundary_t']=={2:499,3:249,4:-1}[r['stage']] for r in inp)
  assert all(r['boundary_vs_final_rms']==0 for r in inp if r['stage']==4)
  ck=list((model/'weights').glob('*/complete.json'));assert len(ck)==1
  assert read(ck[0])['student_updates']==160
  import torch
  from diffusers import UNet2DConditionModel
  state=torch.load(ck[0].parent/'student_rank0.pt',map_location='cpu',mmap=True,weights_only=True)
  with torch.device('meta'):net=UNet2DConditionModel.from_config(read(root/'models/sdxl-base-1.0/unet/config.json'))
  net.load_state_dict(state,strict=True,assign=True);assert all(torch.isfinite(v).all() for v in state.values())
  final=list((model/'eval').glob('*_g000160_*'));assert final
  e=read(final[-1]/'rank0.json');assert len(e['samples'])==8 and all((final[-1]/x['image']).exists() for x in e['samples'])
  (job/'complete.json').write_text(json.dumps({'G':160,'fake':320,'checkpoint':str(ck[0].parent),'cpu_strict_load':True,'COCO10K':'not queued'},indent=2))
  print('COMPLETE; no subsequent evaluation or training queued',flush=True)
if __name__=='__main__':main()
