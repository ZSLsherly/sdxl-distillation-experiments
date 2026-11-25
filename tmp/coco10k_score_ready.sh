set -eu
source /data5/shenglin.zeng/env.sh
cd /data5/shenglin.zeng/sdxl_crossdistill
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 python -u - <<'PY'
import sys, json, time
import numpy as np
sys.path.insert(0,'sdxl_coco10k')
import evaluate as e
def load(method):
    chunks=[]
    for p in sorted((e.OUT/'features'/method).glob('*.npz')):
        with np.load(p) as z:chunks.append({k:z[k].copy() for k in z.files})
    if not chunks:return None
    ids=np.concatenate([x['indices'] for x in chunks])
    if not np.array_equal(ids,np.arange(10000)):return None
    stats={}
    for kind in ['full','patch']:
        f=np.concatenate([x[kind] for x in chunks]).astype(np.float64)
        assert f.shape==(10000,2048) and np.isfinite(f).all()
        stats[kind]=(f.mean(0),np.cov(f,rowvar=False))
    return stats
real=load('real')
assert real is not None
path=e.OUT/'scores_ready.json'
scores=json.loads(path.read_text()) if path.exists() else {}
for method in e.METHODS:
    if method in scores:continue
    stats=load(method)
    if stats is None:
        print('NOT_READY',method,flush=True);continue
    score={'n':10000}
    for kind,key in [('full','FID'),('patch','Patch_FID')]:
        value=float(e.frechet_distance(*real[kind],*stats[kind]))
        assert np.isfinite(value) and value>=-.01
        score[key]=value
        print(method,key,value,flush=True)
    score['computed_unix']=time.time()
    scores[method]=score
    e.write_json(path,scores)
print('READY_SCORES',json.dumps(scores),flush=True)
PY
