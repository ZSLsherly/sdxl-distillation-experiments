import os,sys,json,subprocess,time,fcntl
from pathlib import Path
out=Path.cwd()
lock=(out/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
def query(kind,fields):
 return subprocess.check_output(['nvidia-smi','--query-'+kind+'='+fields,'--format=csv,noheader,nounits'],text=True).splitlines()
def free():
 busy={l.split(',')[0].strip() for l in query('compute-apps','gpu_uuid,pid')}
 return [u.strip() for l in query('gpu','uuid,memory.free,utilization.gpu') for u,m,v in [l.split(',')] if u.strip() in ("GPU-c374184a-ab5a-681a-ed48-025bed2da39b","GPU-baca962f-0eff-b198-1677-8c28d525ecc1") and u.strip() not in busy and int(m)>42000 and int(v)<=3][:1]
for arm in ('sdxl','dmd2'):
 if (out/(arm+'_results.json')).exists():continue
 if arm=='dmd2':
  while not (out/'assets.ready').exists():time.sleep(8)
 folder=out/'images'/arm;folder.mkdir(exist_ok=True,parents=True)
 while len(list(folder.glob('*.png')))<896:
  cards=free()
  if not cards:time.sleep(20);continue
  print('GENERATE',arm,cards,flush=True)
  ps=[]
  for i,c in enumerate(cards):
   with (out/f'{arm}_{i}.log').open('a') as log:
    ps.append(subprocess.Popen([sys.executable,'-u','generate.py',arm,str(i),str(len(cards))],env={**os.environ,'CUDA_VISIBLE_DEVICES':c},stdout=log,stderr=subprocess.STDOUT))
  statuses=[p.wait() for p in ps]
  for i,rc in enumerate(statuses):
   if rc and 'Another compute process' not in (out/f'{arm}_{i}.log').read_text()[-300:]:raise RuntimeError(f'{arm} worker {i}: {rc}')
 while not (out/'assets.ready').exists():time.sleep(8)
 print('SCORE',arm,flush=True)
 subprocess.run([sys.executable,'-u','score.py',arm],env={**os.environ,'CUDA_VISIBLE_DEVICES':''},check=True)
print('COMPLETE',flush=True)
