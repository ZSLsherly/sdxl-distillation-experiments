"""Single-A800 B then C, fresh initialization per arm, final inference weights only."""
import os,json,time,math,shutil,subprocess,fcntl
from pathlib import Path
from launch import command,gpu_check
from turning_batch64_workflow import descendants,foreign_on,stop_own
ROOT=Path(os.environ.get('SDXL_ROOT','/root/sdxl_research'))
CODE=Path(__file__).resolve().parent
OUT=ROOT/'outputs/turning_BC_matched_quota_20261101'

def verify(path):
 status=json.loads((path/'status.json').read_text())
 assert (status['status'],status['student_updates'],status['fake_updates'],status['effective_batch'])==('complete',160,320,64)
 assert status['training_state'] is None
 entries=[json.loads(line) for line in (path/'rank0.jsonl').read_text().splitlines()]
 gs=[r for r in entries if r['phase']=='student_done'];fs=[r for r in entries if r['phase']=='fake']
 assert len(gs)==160 and len(fs)==320
 assert all(math.isfinite(r[k]) for r in gs for k in ['grad_norm','pcm_loss','middle_dm_loss','terminal_dm_loss'])
 assert all(math.isfinite(r[k]) for r in fs for k in ['fake_loss','fake_grad_norm'])
 weights=list((path/'weights').glob('*/complete.json'));assert len(weights)==1
 meta=json.loads(weights[0].read_text());assert meta['student_updates']==160 and meta['world_size']==1
 assert (weights[0].parent/'student_rank0.pt').stat().st_size>4*1024**3
 for g in [0,160]:
  dirs=list((path/'eval').glob(f'*_g{g:06d}_*'));assert dirs
  e=json.loads((dirs[-1]/'rank0.json').read_text());assert len(e['samples'])==8
  assert all((dirs[-1]/s['image']).is_file() and math.isfinite(s['latent_std']) for s in e['samples'])
 return str(weights[0].parent)

def main():
 OUT.mkdir(parents=True,exist_ok=True)
 lock=(OUT/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',TORCHDYNAMO_DISABLE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
 completed={}
 for arm in ['B','C']:
  path=OUT/arm
  if path.exists():completed[arm]=verify(path);continue
  if (OUT/'STOP_REQUEST').exists() or (OUT/'USER_STOP').exists():raise RuntimeError('Stop requested')
  if shutil.disk_usage(OUT).free<12*1024**3:raise RuntimeError('Need 12GiB free before starting next arm')
  for attempt in range(6):
   try:cards=gpu_check([0],76);break
   except RuntimeError:
    if attempt==5:raise
    time.sleep(5)
  config=json.loads((CODE/f'configs/turning_{arm}_quota_a800_g160.json').read_text())
  assert config['match_fake_prior']
  assert config['checkpoint_mode']=='final-weights' and config['save_every_updates']==0
  assert not config.get('resume') and not config.get('init_student') and not config['benchmark']
  uuids={r[1] for r in cards}
  with (OUT/f'{arm}.log').open('x') as log:
   proc=subprocess.Popen(command(config,ROOT,f'{OUT.name}/{arm}',1),cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   print(json.dumps({'event':'started','arm':arm,'pid':proc.pid,'gpus':cards}),flush=True)
   try:
    while proc.poll() is None:
     if foreign_on(uuids,descendants(proc.pid)) or (OUT/'USER_STOP').exists():raise RuntimeError('Foreign GPU process or immediate user stop')
     time.sleep(10)
    if proc.returncode:raise RuntimeError(f'{arm} failed with exit {proc.returncode}; inspect log')
   except BaseException:stop_own(proc);raise
  completed[arm]=verify(path)
  print(json.dumps({'event':'verified','arm':arm,'final_weights':completed[arm]}),flush=True)
 (OUT/'queue_complete.json').write_text(json.dumps(completed,indent=2))
 print('QUEUE_COMPLETE',flush=True)

if __name__=='__main__':main()
