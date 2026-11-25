set -euo pipefail
root=/data1/shenglin.zeng/teacher_turning_20261026
cd "$root"
python3 - <<'PY'
from pathlib import Path
import json,math
p=Path('results512');files=sorted(p.glob('base_dt10_bf16_*.json'))
print('COMPLETE_CASES',len(files))
seen=set();batches={}
for f in files:
 d=json.loads(f.read_text());key=(d['prompt_index'],d['seed']);assert key not in seen;seen.add(key)
 assert len(d['rows'])==100 and len(d['grid'])==101
 for row in d['rows']:
  for field in row['fields'].values():
   assert field['valid'] and all(math.isfinite(v) for v in field.values() if isinstance(v,float))
 batches.setdefault(d['worker'],[]).append({'seconds':d['batch_seconds'],'peak_GiB':d['peak_allocated_GiB']})
summary={'completed':len(files),'passed':bool(files),'workers':{k:{'completed':len(v),'first_batch_seconds':v[0]['seconds'],'peak_GiB':max(x['peak_GiB'] for x in v)} for k,v in batches.items()}}
(p/'validation_snapshot.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
PY
