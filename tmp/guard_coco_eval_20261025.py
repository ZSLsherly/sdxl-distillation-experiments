"""Run independent evaluation only on idle devices; yield if another GPU user appears."""
import os,json,time,subprocess,signal,hashlib
from pathlib import Path
ROOT=Path('/data1/shenglin.zeng/sdxl_eval_uniform_20261025')
OUT=ROOT/'outputs/coco10k';PY=ROOT/'env/bin/python'
GPUS=['GPU-4fba4d7f-66e5-aa77-7210-3803fe7190df','GPU-7328ece9-841e-484d-9e72-d5b762fdce95','GPU-c374184a-ab5a-681a-ed48-025bed2da39b','GPU-baca962f-0eff-b198-1677-8c28d525ecc1']
def gpu_processes():
 s=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
 return [(a.strip(),int(b.strip())) for line in s.splitlines() if ',' in line for a,b in [line.split(',')]]
def record(event,**kw):
 row={'time':time.time(),'event':event,**kw}
 with (OUT/'guard.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
 print(json.dumps(row),flush=True)
env=os.environ.copy();env.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HOME=str(ROOT/'cache'),TOKENIZERS_PARALLELISM='false',LD_LIBRARY_PATH=str(ROOT/'runtime/lib')+':'+env.get('LD_LIBRARY_PATH',''))
workers=[]
try:
 occupied=[(g,p) for g,p in gpu_processes() if g in GPUS]
 if occupied:raise RuntimeError(f'Requested GPUs now occupied: {occupied}')
 raw=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
 for line in raw.splitlines():
  g,mem,util=[s.strip() for s in line.split(',')]
  if g in GPUS and (int(mem)>100 or int(util)>5):raise RuntimeError(f'GPU not idle: {line}')
 for rank,gpu in enumerate(GPUS):
  if any(g==gpu for g,p in gpu_processes()):raise RuntimeError(f'GPU occupied before launch: {gpu}')
  log=(OUT/f'worker_{rank}.log').open('a')
  cmd=['nice','-n','10',str(PY),'-u',str(ROOT/'code/evaluate.py'),'worker','--root',str(ROOT),'--out',str(OUT),'--methods','joint4_uniform','--rank',str(rank),'--world',str(len(GPUS)),'--batch','4']
  p=subprocess.Popen(cmd,cwd=ROOT,env={**env,'CUDA_VISIBLE_DEVICES':gpu},stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  workers.append((gpu,p,log));record('worker_started',rank=rank,pid=p.pid,gpu=gpu)
 while any(p.poll() is None for _,p,_ in workers):
  active={g:p.pid for g,p,_ in workers if p.poll() is None}
  foreign=[(g,pid) for g,pid in gpu_processes() if g in active and pid!=active[g]]
  if foreign:raise RuntimeError(f'Other task appeared on evaluation GPU: {foreign}')
  failed=[p.returncode for _,p,_ in workers if p.poll() not in (None,0)]
  if failed:raise RuntimeError(f'Evaluation worker failed: {failed}')
  time.sleep(10)
 if any(p.returncode for _,p,_ in workers):raise RuntimeError('Worker exit failure')
 record('all_workers_complete')
 inventory=json.loads((OUT/'resume_inventory_20261025.json').read_text())
 for name,meta in inventory['blocks'].items():
  path=OUT/'features/joint4_uniform'/name
  assert hashlib.sha256(path.read_bytes()).hexdigest()==meta['sha256'],name
  assert path.stat().st_mtime_ns==meta['mtime_ns'],name
 record('existing_blocks_unchanged',blocks=len(inventory['blocks']),samples=inventory['completed'])
 with (OUT/'metrics.log').open('a') as log:
  subprocess.run(['nice','-n','19',str(PY),'-u',str(ROOT/'code/evaluate.py'),'metrics','--root',str(ROOT),'--out',str(OUT)],cwd=ROOT,env={**env,'CUDA_VISIBLE_DEVICES':''},stdout=log,stderr=subprocess.STDOUT,check=True)
 scores=json.loads((ROOT/'reference/old_scores.json').read_text());scores.update(json.loads((OUT/'scores.json').read_text()))
 (OUT/'combined_scores.json').write_text(json.dumps(scores,indent=2));record('complete',scores=scores)
except BaseException as exc:
 record('stopped',reason=str(exc))
 for _,p,_ in workers:
  if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
 for _,p,_ in workers:
  try:p.wait(timeout=20)
  except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
 raise
finally:
 for _,_,log in workers:log.close()
