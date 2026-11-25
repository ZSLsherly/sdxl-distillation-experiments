import os,json,subprocess,sys,hashlib
from pathlib import Path
arm='joint4_uniform';out=Path.cwd();base=out.parent.parent
cases=json.loads((out/'cases.json').read_text())
m=json.loads((out.parent/'coco10k/manifest.json').read_text())
assert len(set(r['index'] for r in cases))==1024
assert m['visual_indices']==[r['index'] for r in cases[:128]]
assert all(all(m['samples'][r['index']][k]==r[k] for k in ('seed','prompt')) for r in cases)
rec=m['models'][arm]
assert rec['student_updates']==160 and rec['config']['schedule']==[999,749,499,249,-1]
def query(what,fields):
 return subprocess.check_output(['nvidia-smi','--query-'+what+'='+fields,'--format=csv,noheader,nounits'],text=True).splitlines()
def check(uuid):
 for row in query('compute-apps','gpu_uuid,pid'):
  u,p=map(str.strip,row.split(','))
  if u==uuid and int(p)!=os.getpid():raise RuntimeError('Foreign process on '+u)
if sys.argv[1]=='check':
 for r in rec['shards']:
  with (Path(rec['checkpoint'])/r['name']).open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==r['sha256']
 for r in cases[:128]:assert (out.parent/'coco10k/images'/arm/f"{r['index']:05d}.png").is_file()
 print('verified checkpoint and cases',flush=True)
else:
 uuid=os.environ['CUDA_VISIBLE_DEVICES'];check(uuid)
 import evaluation_core as core
 core.ROOT=base;core.BASE=base/'models/sdxl-base-1.0';core.RUNS={arm:rec}
 pipe=core.load_pipeline(arm,m)
 for j in range(128+4*int(sys.argv[1]),1024,4*int(sys.argv[2])):
  rows=cases[j:j+4]; paths=[out/'images'/f"{r['index']:05d}.png" for r in rows]
  if all(p.exists() for p in paths):continue
  check(uuid)
  ims,_=core.generate(pipe,arm,[r['prompt'] for r in rows],[r['seed'] for r in rows])
  for im,p in zip(ims,paths):
   tmp=p.with_suffix('.partial');im.save(tmp,format='PNG',compress_level=1);tmp.replace(p)
 print('generation complete',flush=True)
