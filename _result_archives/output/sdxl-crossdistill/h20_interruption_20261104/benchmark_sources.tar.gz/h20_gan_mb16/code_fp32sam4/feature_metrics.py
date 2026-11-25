"""Unbiased polynomial KID on the same saved Inception features; FID stays in evaluate.py.

Repeated subsets measure estimator variability, not training-seed uncertainty.
"""
import argparse,json
from pathlib import Path
import numpy as np

def load(path, kind):
    chunks=[]
    for p in sorted(path.glob('*.npz')):
        with np.load(p) as z:chunks.append((z['indices'].copy(),z[kind].copy()))
    if not chunks:raise ValueError(f'No features in {path}')
    ids=np.concatenate([x[0] for x in chunks]);features=np.concatenate([x[1] for x in chunks])
    if len(np.unique(ids))!=len(ids) or not np.isfinite(features).all():raise ValueError('Duplicate or invalid features')
    order=np.argsort(ids);return ids[order],features[order]

def kid(x,y,subset=1000,repeats=20,seed=20261022):
    n=min(subset,len(x),len(y))
    if n<2 or repeats<1:raise ValueError('KID needs >=2 samples and >=1 repeats')
    rng=np.random.default_rng(seed);values=[]
    for _ in range(repeats):
        a=x[rng.choice(len(x),n,replace=False)].astype(np.float64)
        b=y[rng.choice(len(y),n,replace=False)].astype(np.float64)
        aa=(a@a.T/a.shape[1]+1)**3;bb=(b@b.T/b.shape[1]+1)**3;ab=(a@b.T/a.shape[1]+1)**3
        values.append(float((aa.sum()-np.trace(aa)+bb.sum()-np.trace(bb))/(n*(n-1))-2*ab.mean()))
    return {'KID_raw':float(np.mean(values)),'subset_std':float(np.std(values,ddof=1)) if repeats>1 else 0,
            'subset_size':n,'repeats':repeats,'n_real':len(x),'n_generated':len(y)}

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--real',type=Path,required=True)
    ap.add_argument('--generated',type=Path,required=True);ap.add_argument('--repeats',type=int,default=20)
    ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();result={}
    for kind in ('full','patch'):
        ri,x=load(a.real,kind);gi,y=load(a.generated,kind)
        if not np.array_equal(ri,gi):raise ValueError('Real/generated index sets differ')
        result[kind]=kid(x,y,repeats=a.repeats)
    a.out.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
