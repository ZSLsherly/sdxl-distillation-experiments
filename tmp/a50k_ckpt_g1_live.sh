cd /data5/shenglin.zeng/a50k_20261124
python3 - <<'PY'
from pathlib import Path
from collections import defaultdict
import json
d=Path('outputs/first_update_mb16')
for root in ['weights','training_state']:
 x=list((d/root).glob('*/*'))
 print(root, len(x),sum(i.stat().st_size for i in x)/2**30,[i.name for i in x if i.name=='complete.json'])
 print(sorted((p.name,round(p.stat().st_size/2**30,2)) for p in x)[-10:])
seen=defaultdict(list)
for p in d.glob('rank*.jsonl'):
 for l in p.read_text().splitlines():
  r=json.loads(l)
  if 'original_indices_rank' in r:seen[(r['phase'],r.get('g'),r.get('fake_updates',r.get('fake_update',r.get('f'))))].extend(r['original_indices_rank'])
print('groups',[(k,len(v),len(set(v)),v[:3]) for k,v in seen.items()])
PY
tail -3 logs/queue.log
