set -eu
r=/data1/shenglin.zeng/high30_parallel_eval_20261114
p=/data1/shenglin.zeng/sdxl_eval_uniform_20261025/env/bin/python
cd "$r"
tail -n 12 queue.log
"$p" - <<'PY'
from pathlib import Path
import json,sys
sys.path.insert(0,'code')
from resource_guard import status
r=Path('.');out=r/'results/A_clean_global_high30_v2'
for p in sorted(out.glob('status_*.json')):print(p.name,p.read_text())
for p in sorted(out.glob('worker_*.log')):
 print('LOG',p.name,'\n'+'\n'.join(p.read_text(errors='replace').splitlines()[-3:]))
for u in json.loads((r/'node.json').read_text())['gpus']:
 ps,free,util=status(u);print('GPU',u,sorted(ps),round(free/2**30,1),util)
print('FEATURE_BLOCKS',len(list((out/'features/A_clean_global_high30_v2').glob('*.npz'))))
if (out/'all_complete.json').exists():print((out/'all_complete.json').read_text())
PY
