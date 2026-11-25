set -eu
date -Is
/root/sdxl_research/env/bin/python - <<'PY'
from pathlib import Path
import json
from datetime import datetime
root=Path('/root/sdxl_research/outputs')
results={}
for f in root.rglob('rank0.jsonl'):
 if 'source' in f.parts:continue
 rows=[json.loads(s) for s in f.read_text().splitlines() if s.endswith('}')]
 gs=[r for r in rows if r.get('phase')=='student_done']
 if not gs:continue
 d=f.parent
 def t(r):return datetime.strptime(r['timestamp'],'%Y%m%dT%H%M%S%z')
 cfg=json.loads((d/'config.json').read_text())
 state=json.loads((d/'status.json').read_text()) if (d/'status.json').exists() else {}
 results[str(d.relative_to(root))]={'last_g':gs[-1]['g'],'status':state.get('status'),'last_time':gs[-1]['timestamp'],'wall_hours':((t(gs[-1])-t(gs[0])).total_seconds()+gs[0]['cycle_seconds'])/3600,'config':{k:cfg.get(k) for k in ('pcm_weight','terminal_only_dm','fake_updates','both_middle','fake_student_mix','turning_profile','objective','benchmark','init_student')},'weight_files':len(list(d.glob('weights/*/student_rank*.pt')))}
print(json.dumps(results))
PY
ps -eo pid,etime,args | grep -E '(train.py|torchrun|evaluate.py|queue)' | grep -v grep | head -10
