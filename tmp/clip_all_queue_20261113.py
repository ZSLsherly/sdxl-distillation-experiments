"""One independent queue per explicitly allowed physical GPU; resumable blocks."""
import sys,os,time,fcntl,subprocess,json,signal
from pathlib import Path
ROOT=Path('/data3/shenglin.zeng/clip_missing_20261113')
GPUS={'5':'GPU-5eceb5c0-44bb-9b56-d40e-2846ce7ff00f'}
gpu=GPUS[sys.argv[1]]
lock=(ROOT/f'queue_{sys.argv[1]}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
arms=['dmd_only','gan4','joint4','joint34','DM234_fake3','DM1_mid4_fake3','DM1_uniform','H20_extra_GAN','A_pcm10_patchgan','A_mid_clean_x0','A_pcm10_mix1','normalized_u_g160','raw_angle_g160']
def apps():
 out=subprocess.check_output(['nvidia-smi','-i',gpu,'--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True,timeout=15)
 return {int(x) for x in out.splitlines() if x.strip().isdigit()}
env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
while True:
 subprocess.run([sys.executable,str(ROOT/'code/report.py')],check=True)
 if not (ROOT/'validation.json').exists():
  if not (ROOT/'assets_ready.json').exists():time.sleep(15);continue
  subprocess.run([sys.executable,str(ROOT/'code/worker.py'),'validate'],env=dict(env,CUDA_VISIBLE_DEVICES=''),check=True)
 if all((ROOT/'arms'/a/'results.json').exists() for a in arms):print('ALL_COMPLETE',flush=True);break
 if apps():time.sleep(30);continue
 row=subprocess.check_output(['nvidia-smi','-i',gpu,'--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=15).strip().split(',')
 if int(row[0])<21000 or int(row[1])>5:time.sleep(30);continue
 selected=None;handle=None
 for a in arms:
  p=ROOT/'arms'/a
  if not (p/'ready.json').exists() or (p/'results.json').exists():continue
  f=(p/'worker.lock').open('a')
  try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:f.close();continue
  selected=a;handle=f;break
 if selected is None:time.sleep(15);continue
 with (ROOT/'arms'/selected/'worker.log').open('a') as log:
  print('START',selected,gpu,flush=True)
  p=subprocess.Popen([sys.executable,'-u',str(ROOT/'code/worker.py'),selected],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  conflict=False
  while p.poll() is None:
   if apps()-{p.pid}:
    print('RESOURCE_CONFLICT',selected,flush=True);os.killpg(p.pid,signal.SIGTERM);conflict=True;break
   time.sleep(10)
  rc=p.wait()
 handle.close()
 if rc==75:
  print('WAIT_FOR_FREE_GPU',selected,flush=True);time.sleep(30);continue
 if rc and not conflict:
  (ROOT/f'failure_gpu{sys.argv[1]}.json').write_text(json.dumps(dict(arm=selected,returncode=rc)))
  raise RuntimeError(f'{selected} failed; see worker.log')
