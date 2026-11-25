"""One resumable evaluation queue; check selected GPUs only before launch."""
from pathlib import Path
import os,sys,json,subprocess,fcntl,time
import pynvml as nv
r=Path(sys.argv[1]).resolve();job=json.loads((r/'job.json').read_text());py=sys.executable;code=r/'code/evaluate_dmd2.py'
env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONUNBUFFERED='1',MPLCONFIGDIR=str(r/'cache/matplotlib'))

def idle(uid):
 nv.nvmlInit()
 try:
  h=nv.nvmlDeviceGetHandleByUUID(uid)
  assert not nv.nvmlDeviceGetComputeRunningProcesses(h),'Selected GPU occupied'
  assert nv.nvmlDeviceGetMemoryInfo(h).free>30*1024**3 and nv.nvmlDeviceGetUtilizationRates(h).gpu<=5
 finally:nv.nvmlShutdown()

def run(action,name,gpu='',extra=[]):
 if gpu:idle(gpu)
 with (r/'logs'/name).open('a') as log:
  p=subprocess.Popen([py,'-u',str(code),action,'--root',str(r),*extra],cwd=r/'code',env={**env,'CUDA_VISIBLE_DEVICES':gpu},stdout=log,stderr=subprocess.STDOUT)
  rc=p.wait()
 if rc:raise RuntimeError(f'{name} failed with {rc}')

with (r/'queue.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);(r/'logs').mkdir(exist_ok=True)
 if (r/'complete.json').exists():print('ALREADY_COMPLETE');sys.exit(0)
 run('prepare','prepare.log')
 run('probe','probe.log',job['gpu_uuids'][0])
 processes=[];logs=[]
 for rank,uid in enumerate(job['gpu_uuids']):
  idle(uid);log=(r/'logs'/f'worker_{rank}.log').open('a');logs.append(log)
  p=subprocess.Popen([py,'-u',str(code),'worker','--root',str(r),'--rank',str(rank),'--world',str(len(job['gpu_uuids']))],cwd=r/'code',env={**env,'CUDA_VISIBLE_DEVICES':uid},stdout=log,stderr=subprocess.STDOUT);processes.append(p)
  print('WORKER_STARTED',rank,p.pid,uid,flush=True)
 statuses=[p.wait() for p in processes]
 for log in logs:log.close()
 if any(statuses):raise RuntimeError(f'Worker failure {statuses}; completed chunks kept for resumption')
 run('metrics','metrics.log')
 print('ALL_COMPLETE',time.time(),flush=True)
