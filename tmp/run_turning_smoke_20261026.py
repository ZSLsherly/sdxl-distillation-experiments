"""Bounded single-trajectory test on a lightly used GPU; only stop our child."""
import os,sys,time,json,subprocess,signal
from pathlib import Path
root=Path('/data1/shenglin.zeng/teacher_turning_20261026')
out=root/'smoke_gpu0';out.mkdir(exist_ok=True)
def gpu():
 s=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
 return [int(x.strip()) for x in s.strip().split(',')]
free,util=gpu()
assert free>19000 and util<=10,(free,util)
assert not (out/'worker0_complete.json').exists(),'Already completed; do not rerun'
env={**os.environ,'CUDA_VISIBLE_DEVICES':'0','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2',
     'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','TMPDIR':str(root/'tmp'),'HF_HOME':str(root/'cache')}
start=time.time();reason=None;busy=0;samples=[]
with (out/'worker.log').open('a') as log:
 p=subprocess.Popen([sys.executable,'-u',str(root/'sdxl_turning_resample.py'),'--root',str(root),'--out',str(out),'--smoke'],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 print('SMOKE_PID',p.pid,flush=True)
 try:
  while p.poll() is None:
   free,util=gpu()
   mon=subprocess.check_output(['nvidia-smi','pmon','-i','0','-c','1','-s','um'],text=True,timeout=10)
   foreign=[]
   for line in mon.splitlines():
    v=line.split()
    if not v or v[0].startswith('#') or len(v)<5:continue
    if v[1].isdigit() and int(v[1])!=p.pid:
     foreign.append({'pid':int(v[1]),'sm':int(v[3]) if v[3].isdigit() else None})
   busy=busy+1 if any((r['sm'] or 0)>20 for r in foreign) else 0
   row={'elapsed':round(time.time()-start,1),'free_MiB':free,'gpu_util':util,'foreign':foreign}
   samples.append(row);print(json.dumps(row),flush=True)
   if free<3000 or busy>=2 or time.time()-start>600:
    reason='low free memory / sustained foreign load / time limit';break
   time.sleep(4)
 finally:
  if p.poll() is None:
   os.killpg(p.pid,signal.SIGTERM)
   try:p.wait(timeout=10)
   except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
 result={'returncode':p.returncode,'stop_reason':reason,'wall_seconds':time.time()-start,'samples':samples}
 (out/'monitor.json').write_text(json.dumps(result,indent=2))
 print('SMOKE_END',p.returncode,reason,flush=True)
 if p.returncode:print((out/'worker.log').read_text()[-6000:],flush=True)
