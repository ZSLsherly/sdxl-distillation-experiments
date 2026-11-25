set -eu
date '+%F %T %Z'
cd /root/sdxl_research/outputs
/root/sdxl_research/env/bin/python - <<'PY'
import glob,json,os,statistics
from pathlib import Path
for p in sorted(Path('.').iterdir()):
 if not (p.name.startswith(('h20_C_mix1','c_mix1_dm234_fake3','c_mix1_first_dm','c_mix1_dm_pair','c_mix1_pcm_dm4'))): continue
 print('RUN',p)
 for f in p.rglob('rank0.jsonl'):
  rows=[json.loads(s) for s in f.read_text().splitlines() if s.endswith('}')]
  gs=[r for r in rows if r.get('phase')=='student_done']
  if gs:
   tail=gs[-5:]; print('G',gs[-1]['g'],'recent_s',round(statistics.mean(r['cycle_seconds'] for r in tail),1),'at',gs[-1].get('timestamp'))
 for f in list(p.rglob('status.json'))+list(p.rglob('status_0.json'))+list(p.rglob('complete.json'))+list(p.rglob('comparison.json')):
  if any(k in f.parts for k in ('weights','source','eval')):continue
  j=json.loads(f.read_text());print(f, {k:j[k] for k in ('status','student_updates','fake_updates','D_updates','phase','done','method','scores','target','updated_unix') if k in j})
 print('eval_g', [d.name.split('_')[1] for d in sorted(p.glob('eval/*'))[-2:]])
PY
ps -eo pid,stat,comm,args | grep -E '(after_coco.py|after_g160.py|queue_dm_pair_coco|pcm_dm4|run_first_dm)' | grep -v grep | cut -c1-150 | head -12
for f in /root/autodl-tmp/h20_direct_after_coco_20261103/queue.log /root/autodl-tmp/h20_coco_after_g160/assessment.log; do if test -f "$f";then tail -1 "$f";fi;done


