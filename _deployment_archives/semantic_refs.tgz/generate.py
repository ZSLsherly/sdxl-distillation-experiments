import sys,os,json,subprocess
from pathlib import Path
import evaluation_core as core
arm,rank,world=sys.argv[1:];rank=int(rank);world=int(world)
assert arm in ('sdxl','dmd2')
out=Path.cwd();base=out.parent.parent
uuid=os.environ['CUDA_VISIBLE_DEVICES']
if uuid not in ('GPU-c374184a-ab5a-681a-ed48-025bed2da39b','GPU-baca962f-0eff-b198-1677-8c28d525ecc1'):raise SystemExit('outside designated GPUs 3 and 6')
def safe():
    lines=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True).splitlines()
    for line in lines:
        u,p=line.split(',')
        if u.strip()==uuid and int(p)!=os.getpid():raise RuntimeError('Another compute process on GPU')
safe()
core.ROOT=out;core.BASE=base/'models/sdxl-base-1.0';core.RUNS={}
m=json.loads((out/'ref.json').read_text());cases=json.loads((out/'cases.json').read_text())
pipe=core.load_pipeline(arm,m)
if arm=='sdxl':assert pipe.scheduler.__class__.__name__==m['models']['sdxl']['scheduler']['_class_name']
bs=2 if arm=='sdxl' else 4
for start in range(128+bs*rank,1024,bs*world):
    rows=cases[start:start+bs]
    dest=[out/'images'/arm/f"{r['index']:05d}.png" for r in rows]
    if all(p.exists() for p in dest):continue
    safe()
    ims,_=core.generate(pipe,arm,[r['prompt'] for r in rows],[r['seed'] for r in rows])
    for im,p in zip(ims,dest):
        t=p.with_suffix('.part');im.save(t,format='PNG',compress_level=1);t.replace(p)
print('DONE',arm,rank,flush=True)
