date -Is
. /root/sdxl_research/env.sh
python - <<'PY'
from pathlib import Path
import json
p=Path('/root/autodl-tmp/sdxl_runs/outputs/a800_B_batch_probe_20261101')
if (p/'results.json').exists():
 for r in json.loads((p/'results.json').read_text()):
  print({k:r.get(k) for k in ['micro_batch','accumulation','status','stable_cycle_seconds']})
for q in sorted(p.glob('mb*'),key=lambda x:x.stat().st_mtime):
 if q.is_dir():
  f=q/'rank0.jsonl'
  if f.exists():
   rows=[json.loads(x) for x in f.read_text().splitlines()];updates=[x for x in rows if x.get('phase') in ('fake','student_done')]
   if updates:print(q.name,[{k:x.get(k) for k in ['phase','g','fake_update','seconds','cycle_seconds','peak_allocated_GiB','grad_norm']} for x in updates[-2:]])
logs=sorted(p.glob('mb*.log'),key=lambda x:x.stat().st_mtime)
if logs:print('LATEST_LOG',logs[-1].name,'\n'+'\n'.join(logs[-1].read_text(errors='replace').splitlines()[-5:]))
if (p/'finished.json').exists(): print('PROBE_FINISHED')
PY
nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits
