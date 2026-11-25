set -eu
. /root/sdxl_research/env.sh
python - <<'PY'
import os,json,time,statistics,datetime
from pathlib import Path
root=Path(os.environ['SDXL_ROOT'])/'outputs'
print('now',datetime.datetime.now().astimezone().isoformat(timespec='seconds'))
for n in ['c_mix1_dm234_fake3_20261103/model','bc_winner_late_mix_20261102/late_mix']:
 p=root/n;j=json.loads((p/'status.json').read_text());rows=[json.loads(l) for l in (p/'rank0.jsonl').read_text().splitlines() if l.endswith('}')]
 g=[r for r in rows if r['phase']=='student_done'];f=[r for r in rows if r['phase']=='fake']; print(n,'state',j['status'],'G',len(g),'fake',len(f),'phase',j.get('phase'))
 print('cycles',[round(r['cycle_seconds'],1) for r in g[-6:]],'fake seconds',[round(r['seconds'],1) for r in f[-6:]],'G seconds',[round(r['seconds'],1) for r in g[-6:]])
 def epoch(r):return datetime.datetime.strptime(r['timestamp'],'%Y%m%dT%H%M%S%z').timestamp()
 print('wall inter-G',[(r['g'],round(epoch(r)-epoch(a),1)) for a,r in zip(g[-7:-1],g[-6:])] if len(g)>=7 else [round(epoch(r)-epoch(a),1) for a,r in zip(g,g[1:])])
 print('wall per G',(epoch(g[-1])-epoch(g[0]))/(len(g)-1) if len(g)>1 else None)
 if n.startswith('c_mix'):print('tail',[(r['phase'],r.get('fake_stage'),r.get('g'),r['timestamp']) for r in rows[-2:]])
PY
