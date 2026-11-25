set -eu
cd /root/sdxl_research/outputs
/root/sdxl_research/env/bin/python - <<'PY'
from pathlib import Path
from datetime import datetime
import json,statistics
data={}
for pat in ['h20_C_mix1*','c_mix1*','turning_BC_paired_quota_20261102','bc_winner_late_mix_20261102','interval_D_20261103']:
 for d in Path('.').glob(pat):
  for f in d.rglob('rank0.jsonl'):
   rows=[json.loads(l) for l in f.read_text().splitlines() if l.endswith('}')]
   gs=[r for r in rows if r.get('phase')=='student_done']
   if not gs:continue
   def dt(r):return datetime.strptime(r['timestamp'],'%Y%m%dT%H%M%S%z')
   s=gs[0].get('cycle_seconds',0)
   data[str(f.parent)]={'g':gs[-1]['g'],'hours':round(((dt(gs[-1])-dt(gs[0])).total_seconds()+s)/3600,4),'s_per_g':round(statistics.mean(r['cycle_seconds'] for r in gs[-5:]),1),'date':gs[-1]['timestamp']}
  for f in d.rglob('complete.json'):
   j=json.loads(f.read_text())
   if j.get('scores'):data[str(f.parent)+' scores']=j['scores']
print(json.dumps(data))
PY
