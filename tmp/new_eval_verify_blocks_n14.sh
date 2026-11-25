/data3/shenglin.zeng/sdxl_turning_ablation_20261028/env/bin/python - <<'PY'
import json,numpy as np
from pathlib import Path
r=Path('/data3/shenglin.zeng/new_models_eval_20261114');out=r/'results/A_clean_global'
ids=[]
for p in sorted((out/'features/A_clean_global').glob('*.npz')):
 with np.load(p) as z:
  ids.extend(z['indices'].tolist())
  assert z['full'].shape==z['patch'].shape==(len(z['indices']),2048)
  assert np.isfinite(z['full']).all() and np.isfinite(z['patch']).all()
assert ids==list(range(len(ids)))
selected={x['index'] for x in json.loads((r/'selection.json').read_text())['cases']}
count=0
for p in (out/'clip_features').glob('*.npy'):
 z=np.load(p);assert int(p.stem) in selected and z.shape==(512,) and np.isfinite(z).all();count+=1
print('VERIFIED_FID_SAMPLES',len(ids),'VALID_CLIP_FEATURES',count)
print((out/'status_0.json').read_text())
PY
