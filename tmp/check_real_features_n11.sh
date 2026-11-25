r=/data1/shenglin.zeng/coco_missing_20261111
/data1/shenglin.zeng/teacher_turning_20261026/env/bin/python - <<'PY'
from pathlib import Path
import numpy as np,json
r=Path('/data1/shenglin.zeng/coco_missing_20261111')
m=json.loads((r/'reference/manifest.json').read_text());print('samples',m['n'],len(m['samples']),'protocol',m['protocol'])
ids=np.concatenate([np.load(p)['indices'] for p in sorted((r/'reference/features/real').glob('*.npz'))]);print('real_features',len(ids),'unique',len(set(ids.tolist())),'sequential',np.array_equal(ids,np.arange(10000)))
PY
