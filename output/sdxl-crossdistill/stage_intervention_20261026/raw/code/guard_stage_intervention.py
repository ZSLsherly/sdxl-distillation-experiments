import argparse,csv,io,json,os,signal,subprocess,sys,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');a=p.parse_args()
root=Path('/data1/shenglin.zeng/teacher_turning_20261026');out=root/('stage_smoke' if a.smoke else 'stage_intervention')
out.mkdir(exist_ok=True);cards=[1] if a.smoke else [1,4];workers={};logs=[];stable={g:0 for g in cards};calm={g:0 for g in cards}
def query():
 s=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=10)
 return {int(r[0]):{'free':int(r[1]),'util':int(r[2])} for r in csv.reader(io.StringIO(s))}
start=time.time()
try:
 while True:
  q=query()
  for rank,gpu in enumerate(cards):
   if gpu in workers:continue
   stable[gpu]=stable[gpu]+1 if q[gpu]['free']>=18000 and q[gpu]['util']<=15 else 0
   if stable[gpu]<3:continue
   (out/f'PAUSE_{rank}').unlink(missing_ok=True)
   env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2',
        'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','TMPDIR':str(root/'tmp'),'HF_HOME':str(root/'cache')}
   log=(out/f'worker{rank}.log').open('a');logs.append(log)
   cmd=[sys.executable,'-u',str(root/'sdxl_stage_intervention.py'),'--root',str(root),'--out',str(out),
        '--rank',str(rank),'--workers',str(len(cards)),'--limit','1' if a.smoke else '32','--batch','1' if a.smoke else '4']
   workers[gpu]=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   print('START',gpu,workers[gpu].pid,flush=True)
  mon=subprocess.check_output(['nvidia-smi','pmon','-c','1','-s','um'],text=True,timeout=10)
  owned={v.pid for v in workers.values()};other={g:[] for g in cards}
  for line in mon.splitlines():
   v=line.split()
   if len(v)>4 and v[0].isdigit() and v[1].isdigit() and int(v[0]) in other and int(v[1]) not in owned:
    other[int(v[0])].append({'pid':int(v[1]),'sm':int(v[3]) if v[3].isdigit() else None})
  state={}
  for rank,gpu in enumerate(cards):
   proc=workers.get(gpu);flag=out/f'PAUSE_{rank}'
   if proc is None:s='waiting'
   elif proc.poll() is not None:
    if proc.returncode:raise RuntimeError(f'GPU {gpu} failed: {proc.returncode}')
    s='complete'
   else:
    if q[gpu]['free']<3500:raise RuntimeError(f'GPU {gpu} memory reserve reached')
    busy=any((v['sm'] or 0)>20 for v in other[gpu]);calm[gpu]=0 if busy else calm[gpu]+1
    if busy:flag.touch()
    elif calm[gpu]>=3:flag.unlink(missing_ok=True)
    s='yielding' if flag.exists() else 'running'
   state[str(gpu)]={**q[gpu],'state':s,'pid':proc.pid if proc else None,'foreign':other[gpu]}
  (out/'status.json').write_text(json.dumps({'elapsed':time.time()-start,'time':time.time(),'gpus':state,
       'latent_cases':len(list(out.glob('latent_*.pt'))),'pngs':len(list(out.glob('*.png')))},indent=2))
  if len(workers)==len(cards) and all(v.poll()==0 for v in workers.values()):break
  if (out/'STOP_REQUEST').exists():raise RuntimeError('Stop requested')
  if time.time()-start>5400:raise RuntimeError('90 minute limit; completed outputs retained')
  time.sleep(4)
 (out/'complete.json').write_text(json.dumps({'complete':True,'wall_seconds':time.time()-start,'cases':1 if a.smoke else 32}))
 print('COMPLETE',round(time.time()-start,1),flush=True)
except BaseException as e:
 (out/'error.json').write_text(json.dumps({'error':str(e),'time':time.time()}));raise
finally:
 for proc in workers.values():
  if proc.poll() is None:
   os.killpg(proc.pid,signal.SIGTERM)
   try:proc.wait(timeout=10)
   except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
 for log in logs:log.close()
