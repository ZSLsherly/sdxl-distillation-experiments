"""Bounded paired GPU probe only. Does NOT launch G160 automatically."""
import csv,io,json,os,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path('/data1/shenglin.zeng/teacher_turning_20261026');OUT=ROOT/'outputs/turning_boundary_ablation_20261027'
CODE=OUT/'code/sdxl_research';sys.path.insert(0,str(CODE))
from launch import command
GPU=0
def cards():
 s=subprocess.check_output(['nvidia-smi','-i',str(GPU),'--query-gpu=uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=10)
 r=next(csv.reader(io.StringIO(s)));return r[0].strip(),int(r[1]),int(r[2])
def memory():return next(int(x.split()[1])/2**20 for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:'))
def apps(uuid):
 s=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True,timeout=10)
 return {int(r[1]):int(r[2]) for r in csv.reader(io.StringIO(s)) if len(r)>2 and r[0].strip()==uuid}
def tree(pid):
 own={pid};todo=[pid]
 while todo:
  pid=todo.pop()
  try:children=(Path('/proc')/str(pid)/'task'/str(pid)/'children').read_text().split()
  except OSError:continue
  for c in children:
   c=int(c)
   if c not in own:own.add(c);todo.append(c)
 return own
env={**os.environ,'CUDA_VISIBLE_DEVICES':'0','NCCL_P2P_DISABLE':'1','NCCL_IB_DISABLE':'1','OMP_NUM_THREADS':'2',
 'OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2','TORCHDYNAMO_DISABLE':'1','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','TMPDIR':str(ROOT/'tmp')}
try:
 for arm in ['normalized_u','raw_angle']:
  uuid,free,util=cards();assert free>=18000 and util<=5,(free,util)
  assert memory()>=90,'Insufficient CPU RAM for offloaded models and Adam'
  baseline=apps(uuid);config=json.loads((CODE/'configs'/f'turning_{arm}_g160.json').read_text())
  config.update(benchmark=True,benchmark_batch=8,benchmark_cycles=3,benchmark_warmup=1,
      micro_batch=8,rollout_batch=8,max_student_updates=3,hours=1)
  run=f'turning_boundary_ablation_20261027/short_{arm}'
  assert not (ROOT/'outputs'/run).exists(),'Use fresh short-test output'
  cmd=command(config,ROOT,run,1);cmd[5]=str(CODE/'capped_turning_train.py')
  (OUT/f'short_{arm}_launch.json').write_text(json.dumps({'command':cmd,'gpu':GPU,'baseline_processes':baseline,'gpu_allocator_cap_GiB':14,'config':config},indent=2))
  with (OUT/f'short_{arm}.log').open('w') as log:
   proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   start=time.time();print('START',arm,proc.pid,flush=True)
   try:
    while proc.poll() is None:
     own=tree(proc.pid);current=apps(uuid);uuid,free,util=cards()
     new=[pid for pid in current if pid not in own and pid not in baseline]
     grown=[pid for pid,mb in current.items() if pid in baseline and mb>baseline[pid]+512]
     mon=subprocess.check_output(['nvidia-smi','pmon','-i',str(GPU),'-c','1','-s','um'],text=True,timeout=10)
     busy=[]
     for line in mon.splitlines():
      v=line.split()
      if len(v)>4 and v[1].isdigit() and int(v[1]) not in own and v[3].isdigit() and int(v[3])>10:busy.append(int(v[1]))
     if new or grown or busy or free<4000 or memory()<20 or time.time()-start>1200:
      raise RuntimeError(f'Stop own probe: new={new}, memory_growth={grown}, other_active={busy}, free_MiB={free}, CPU_available_GiB={memory():.1f}')
     time.sleep(4)
    if proc.returncode:raise RuntimeError(f'{arm} failed: {proc.returncode}')
   finally:
    if proc.poll() is None:
     os.killpg(proc.pid,signal.SIGTERM)
     try:proc.wait(timeout=15)
     except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
  r=json.loads((ROOT/'outputs'/run/'benchmark.json').read_text());assert r['student_updates']==3 and r['fake_updates']==6
  print('PROBE_COMPLETE',arm,json.dumps(r),flush=True)
 (OUT/'paired_GPU_probe_complete.json').write_text(json.dumps({'complete':True,'formal_training_started':False}))
except BaseException as e:
 (OUT/'paired_GPU_probe_error.json').write_text(json.dumps({'error':str(e),'time':time.time()}));raise
