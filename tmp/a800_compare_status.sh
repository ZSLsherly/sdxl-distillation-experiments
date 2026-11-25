date -Is
. /root/sdxl_research/env.sh
python - <<'PY'
from pathlib import Path
import json
p=Path('/root/autodl-tmp/sdxl_runs/outputs/a800_B_offload_compare_20261101')
if (p/'results.json').exists():
 for r in json.loads((p/'results.json').read_text()):print({k:r.get(k) for k in ['label','micro_batch','status','stable_cycle_seconds']})
for q in p.iterdir():
 if q.is_dir() and (q/'rank0.jsonl').exists():
  rows=[json.loads(s) for s in (q/'rank0.jsonl').read_text().splitlines()]
  rows=[x for x in rows if x['phase'] in ('student_done','fake')]
  print(q.name,'completed_G',[(x['g'],round(x['cycle_seconds'],2),round(x['peak_allocated_GiB'],2)) for x in rows if x['phase']=='student_done'])
  if rows:print(q.name,[{k:x.get(k) for k in ['phase','g','fake_update','cycle_seconds','peak_allocated_GiB','grad_norm']} for x in rows[-2:]])
if (p/'finished.json').exists():print('FINISHED')
PY
nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits
