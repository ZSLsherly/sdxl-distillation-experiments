"""Run independent shards on lightly loaded GPUs; yield compute to existing jobs."""
import argparse,csv,io,json,os,signal,subprocess,sys,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--batch',type=int,required=True);p.add_argument('--gpus',default='0,1,4');a=p.parse_args()
root=Path('/data1/shenglin.zeng/teacher_turning_20261026');out=root/'results512';out.mkdir(exist_ok=True)
cards=[int(v) for v in a.gpus.split(',')];workers={};handles=[];stable={i:0 for i in cards};resume={i:0 for i in cards}
def query():
 raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=10)
 return {int(r[0]):{'uuid':r[1].strip(),'free':int(r[2]),'util':int(r[3])} for r in csv.reader(io.StringIO(raw))}
def stop(proc):
 if proc.poll() is None:
  os.killpg(proc.pid,signal.SIGTERM)
  try:proc.wait(timeout=10)
  except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
start=time.time();status={};error=None
try:
 with (out/'monitor.jsonl').open('a') as monitor:
  while True:
   q=query()
   for rank,gpu in enumerate(cards):
    flag=out/f'PAUSE_{rank}'
    if gpu not in workers:
     stable[gpu]=stable[gpu]+1 if q[gpu]['free']>=18000 and q[gpu]['util']<=15 else 0
     if stable[gpu]<3:continue
     flag.unlink(missing_ok=True)
     env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2',
          'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','TMPDIR':str(root/'tmp'),'HF_HOME':str(root/'cache')}
     log=(out/f'worker{rank}.log').open('a');handles.append(log)
     proc=subprocess.Popen([sys.executable,'-u',str(root/'sdxl_turning_batch.py'),'--root',str(root),'--out',str(out),
        '--batch',str(a.batch),'--rank',str(rank),'--workers',str(len(cards))],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
     workers[gpu]=proc;print('START',gpu,proc.pid,flush=True)
   mon=subprocess.check_output(['nvidia-smi','pmon','-c','1','-s','um'],text=True,timeout=10)
   other={i:[] for i in cards};owned={v.pid for v in workers.values()}
   for line in mon.splitlines():
    v=line.split()
    if len(v)<5 or not v[0].isdigit() or not v[1].isdigit():continue
    gpu,pid=int(v[0]),int(v[1])
    if gpu in other and pid not in owned:other[gpu].append({'pid':pid,'sm':int(v[3]) if v[3].isdigit() else None})
   status={}
   for rank,gpu in enumerate(cards):
    proc=workers.get(gpu);flag=out/f'PAUSE_{rank}'
    if proc is None:state='waiting_for_low_load'
    elif proc.poll() is not None:
     if proc.returncode:raise RuntimeError(f'GPU {gpu} worker failed, exit {proc.returncode}')
     state='complete'
    else:
     if q[gpu]['free']<3500:raise RuntimeError(f'GPU {gpu} free memory below reserve')
     high=any((r['sm'] or 0)>20 for r in other[gpu])
     resume[gpu]=0 if high else resume[gpu]+1
     if high:flag.touch()
     elif resume[gpu]>=3:flag.unlink(missing_ok=True)
     state='yielding_to_other_task' if flag.exists() else 'sampling'
    status[str(gpu)]={**q[gpu],'state':state,'pid':proc.pid if proc else None,'foreign':other[gpu]}
   row={'time':time.time(),'elapsed':round(time.time()-start,1),'gpus':status,'completed_trajectories':len(list(out.glob('base_dt10_bf16_*.json')))}
   monitor.write(json.dumps(row)+'\n');monitor.flush()
   (out/'status.json').write_text(json.dumps(row,indent=2))
   if len(workers)==len(cards) and all(v.poll()==0 for v in workers.values()):break
   if (out/'STOP_REQUEST').exists():raise RuntimeError('Stop requested')
   if time.time()-start>4*3600:raise RuntimeError('Four-hour limit reached; completed cases remain resumable')
   time.sleep(4)
  count=len(list(out.glob('base_dt10_bf16_*.json')));assert count==512,count
  (out/'complete.json').write_text(json.dumps({'complete':True,'trajectories':count,'wall_seconds':time.time()-start}))
except BaseException as e:
 error=str(e);(out/'error.json').write_text(json.dumps({'error':error,'time':time.time()}));raise
finally:
 for proc in workers.values():stop(proc)
 for h in handles:h.close()
