set -euo pipefail
cd /data1/shenglin.zeng/sdxl_eval_uniform_20261025
export CUDA_VISIBLE_DEVICES=''
export LD_LIBRARY_PATH="$PWD/runtime/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
nice -n 19 env/bin/python - <<'PY'
import json,hashlib,time
from pathlib import Path
import numpy as np
root=Path.cwd();out=root/'outputs/coco10k'
for p in Path('/proc').glob('[0-9]*/cmdline'):
 try:cmd=p.read_bytes().replace(b'\0',b' ').decode()
 except (OSError,UnicodeDecodeError):continue
 if (root.as_posix()+'/code/evaluate.py worker' in cmd or 'python -u guard_coco_eval_20261025.py' in cmd) and 'bash -c' not in cmd:
  raise RuntimeError('Existing evaluation process: '+p.parts[2])
manifest=json.loads((out/'manifest.json').read_text())
assert manifest['n']==10000 and manifest['models']['joint4_uniform']['student_updates']==160
valid={f'{l:05d}_{min(l+64,10000):05d}.npz':list(range(l,min(l+64,10000))) for l in range(0,10000,64)}
inventory={};seen=[]
for p in sorted((out/'features/joint4_uniform').glob('*.npz')):
 assert p.name in valid,p.name
 with np.load(p) as z:
  ids=z['indices'].tolist();assert ids==valid[p.name],p.name
  for k in ('full','patch'):assert z[k].shape==(len(ids),2048) and np.isfinite(z[k]).all(),p.name
 seen.extend(ids);inventory[p.name]={'n':len(ids),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'mtime_ns':p.stat().st_mtime_ns}
assert len(seen)==len(set(seen))
record={'checked_unix':time.time(),'completed':len(seen),'remaining':10000-len(seen),'blocks':inventory,'world':4,'reused_by_rank':[sum(v['n'] for k,v in inventory.items() if int(k[:5])//64%4==r) for r in range(4)]}
(out/'resume_inventory_20261025.json').write_text(json.dumps(record,indent=2))
print(json.dumps({k:v for k,v in record.items() if k!='blocks'},indent=2))
print('ALL_EXISTING_BLOCKS_VALID',len(inventory))
PY
