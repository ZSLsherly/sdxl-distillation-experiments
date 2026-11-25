"""Single GPU queue using only one explicitly reserved idle device per host."""
import sys,os,json,time,fcntl,subprocess,signal
from pathlib import Path
ROOT=Path(sys.argv[1]);node=json.loads((ROOT/'node.json').read_text());gpu=node['gpu']
lock=(ROOT/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
def apps():
 s=subprocess.check_output(['nvidia-smi','-i',gpu,'--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True,timeout=15)
 return {int(x) for x in s.splitlines() if x.strip().isdigit()}
env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
while True:
 remaining=[a for a in node['arms'] if not (ROOT/'results'/a/'all_complete.json').exists()]
 if not remaining:print('QUEUE_COMPLETE',flush=True);break
 arm=next((a for a in remaining if (ROOT/'results'/a/'ready.json').exists()),None)
 if arm is None or not (ROOT/'assets_ready.json').exists():time.sleep(20);continue
 if apps():time.sleep(30);continue
 row=subprocess.check_output(['nvidia-smi','-i',gpu,'--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=15).strip().split(',')
 if int(row[0])<20000 or int(row[1])>5:time.sleep(30);continue
 with (ROOT/'results'/arm/'worker.log').open('a') as log:
  print('START',arm,gpu,flush=True)
  p=subprocess.Popen([sys.executable,'-u',str(ROOT/'code/worker.py'),str(ROOT),arm],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  conflict=False
  while p.poll() is None:
   if apps()-{p.pid}:os.killpg(p.pid,signal.SIGTERM);conflict=True;print('YIELD_GPU',arm,flush=True);break
   time.sleep(5)
  rc=p.wait()
 if conflict or rc==75:time.sleep(30);continue
 if rc:raise RuntimeError(f'{arm} failed, see worker.log; completed blocks retained')
