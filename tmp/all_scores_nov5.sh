set -eu
. /root/sdxl_research/env.sh
cd "$SDXL_ROOT/outputs"
date -Iseconds
python - <<'PY'
import json
from pathlib import Path
for d in ('c_mix1_pcm_dm4_matched_time','A_pcm10_g160','dm1_mid4_uniform_mix1_g160'):
 p=Path(d);print(d)
 for f in ('model/status.json','g160_scores.json','comparison.json','clip128.json','queue.log'):
  z=p/f
  if z.exists():
   if f.endswith('log'):print(z.read_text()[-350:]);continue
   v=json.loads(z.read_text());print(f,v.get('metrics',v) if f!='model/status.json' else {k:v.get(k) for k in ('status','student_updates','fake_updates','timestamp')})
 for z in p.glob('*/status_0.json'):
  q=json.loads(z.read_text());print(z.parent.name,q.get('done'))
 if d.startswith('c_mix1'):
  for l in (p/'model/rank0.jsonl').open():
   if 'budget_comparison' in l:print(l[:350])
PY
