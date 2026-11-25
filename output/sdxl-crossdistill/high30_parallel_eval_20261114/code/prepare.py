from pathlib import Path
import hashlib,json,sys,os
import torch,numpy as np
ROOT=Path(sys.argv[1]);old=Path('/data1/shenglin.zeng/sdxl_eval_uniform_20261025')
ARM='A_clean_global_high30_v2';out=ROOT/'results'/ARM
digest='95cf20b296bdc5d5c2fffce717c969f624aca7e5c81bb6c7de3a6ee2245be437'
def write(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2))
def link(dst,src):
    assert src.exists(),src
    dst.parent.mkdir(parents=True,exist_ok=True)
    if dst.is_symlink():assert dst.resolve()==src.resolve()
    else:assert not dst.exists();dst.symlink_to(src,target_is_directory=True)
weight=ROOT/'checkpoint/student_rank0.pt'
assert weight.stat().st_size==5135506134
h=hashlib.sha256()
with weight.open('rb') as f:
    for b in iter(lambda:f.read(16<<20),b''):h.update(b)
assert h.hexdigest()==digest
state=torch.load(weight,map_location='cpu',weights_only=True,mmap=True)
assert len(state)==1680 and all(v.dtype==torch.bfloat16 and torch.isfinite(v).all() for v in state.values())
write(weight.parent/'shapes_rank0.json',{k:dict(shape=list(v.shape),placements=[]) for k,v in state.items()})
del state
link(ROOT/'models/sdxl-base-1.0',old/'models/sdxl-base-1.0')
link(ROOT/'models/metrics',old/'models/metrics')
link(ROOT/'models/clip',old/'outputs/semantic_joint4_uniform_1024/model')
link(out/'features/real',old/'outputs/coco10k/features/real')
m=json.loads((old/'outputs/coco10k/manifest.json').read_text())
cfg=json.loads((ROOT/'training_effective_config.json').read_text());cfg['schedule']=[999,749,499,249,-1]
m['models']={ARM:dict(config=cfg,checkpoint=str(weight.parent),world_size=1,student_updates=160,shards=[dict(name=weight.name,sha256=digest)])}
assert m['n']==len(m['samples'])==10000
write(out/'manifest.json',m)
sel=json.loads((ROOT/'selection.json').read_text())
assert len(sel['cases'])==len({x['index'] for x in sel['cases']})==1024
for r in sel['cases']:
    s=m['samples'][r['index']];assert r==dict(index=s['index'],seed=s['seed'],prompt=s['prompt'])
ids=[]
for p in sorted((out/'features/real').glob('*.npz')):
    with np.load(p) as z:
        ids.extend(z['indices'].tolist())
        assert all(z[k].shape==(len(z['indices']),2048) and np.isfinite(z[k]).all() for k in ('full','patch'))
assert ids==list(range(10000))
import evaluation_core as core
parts=[[i for l,r in core.feature_blocks(10000,rank,6) for i in range(l,r)] for rank in range(6)]
assert sorted(sum(parts,[]))==list(range(10000))
write(ROOT/'assets_ready.json',dict(bytes=weight.stat().st_size,mtime_ns=weight.stat().st_mtime_ns,sha256=digest,
    reference=str(old/'outputs/coco10k/manifest.json'),rank_counts=list(map(len,parts)),shape_metadata='derived from verified full BF16 state_dict'))
print('ASSETS_READY',list(map(len,parts)),flush=True)
