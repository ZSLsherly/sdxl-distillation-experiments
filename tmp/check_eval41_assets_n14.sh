set -eu
b=/data1/shenglin.zeng/sdxl_eval_uniform_20261025
"$b/env/bin/python" - <<'PY'
from pathlib import Path
import json,numpy as np
b=Path('/data1/shenglin.zeng/sdxl_eval_uniform_20261025')
m=json.loads((b/'outputs/coco10k/manifest.json').read_text())
print('REFERENCE',m.keys(),'N',m.get('n'),'SAMPLES',len(m['samples']),'FIRST',m['samples'][0])
blocks=sorted((b/'outputs/coco10k/features/real').glob('*.npz'))
ids=[]
for p in blocks:
 with np.load(p) as z:
  ids.extend(z['indices'].tolist());assert z['full'].shape==(len(z['indices']),2048) and z['patch'].shape==z['full'].shape
  assert np.isfinite(z['full']).all() and np.isfinite(z['patch']).all()
assert ids==list(range(10000))
print('REAL_FEATURES',len(blocks),'VALID10000')
clip=b/'outputs/semantic_joint4_uniform_1024/model'
print('CLIP_FILES',[p.name for p in clip.iterdir()])
print('CLIP_REV',(clip/'revision.txt').read_text())
import transformers,cleanfid,torch,diffusers
print('ENV',torch.__version__,diffusers.__version__,transformers.__version__)
PY
