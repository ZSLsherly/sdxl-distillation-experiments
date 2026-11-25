python3 - <<'PY'
import json,tarfile
from pathlib import Path
b=Path('/data5/shenglin.zeng')
r=json.loads((b/'sdxl_crossdistill/outputs/coco10k_g160_20261020/manifest.json').read_text())
for f,a in [('part_004.tar','A_mid_clean_x0'),('part_015.tar','A_pcm10_patchgan')]:
 with tarfile.open(b/'checkpoints/a800_archive_20261106/evidence'/f) as t:
  for p in t:
   if p.name.endswith('/'+a+'/manifest.json'):
    m=json.load(t.extractfile(p))
    print(a,'sizes',len(m['samples']),len(r['samples']),'visual_equal',m['visual_indices']==r['visual_indices'])
    diffs=[i for i,(x,y) in enumerate(zip(m['samples'],r['samples'])) if x!=y]
    print('sample_diffs',len(diffs))
    for i in diffs[:2]:print(i,m['samples'][i],r['samples'][i])
    print('semantic_identity',all(x['prompt']==y['prompt'] and x['seed']==y['seed'] for x,y in zip(m['samples'],r['samples'])))
    print('visual',m['visual_indices'][:8],r['visual_indices'][:8])
PY
