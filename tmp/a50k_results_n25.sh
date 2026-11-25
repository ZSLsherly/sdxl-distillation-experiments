cd /data5/shenglin.zeng/a50k_20261124 || exit 1
date
df -h /data5 /data6
for f in logs/singlepass_queue.log logs/no_middle_queue.log; do echo "$f";tail -4 "$f";done
python3 - <<'PY'
from pathlib import Path
import json,statistics,math
for name in ('A50K_singlepass','A50K_no_middle_singlepass'):
 d=Path('outputs')/name
 print('RUN',name,flush=True)
 if not d.exists():continue
 p=d/'status.json'
 if p.exists(): print('STATUS',p.read_text()[:1750])
 rows=[]
 for line in (d/'rank0.jsonl').read_text().splitlines() if (d/'rank0.jsonl').exists() else []:
  try:rows.append(json.loads(line))
  except ValueError:pass
 gs=[r for r in rows if r['phase']=='student_done']
 if gs:
  print('LAST',gs[-1]['g'],gs[-1]['timestamp'],'of',len(gs))
  for k in ('pcm_loss','middle_dm_loss','terminal_dm_loss','grad_norm','cycle_seconds','peak_allocated_GiB'):
   print(k,'last',round(gs[-1][k],6),'last20avg',round(statistics.mean(r[k] for r in gs[-20:]),6),'finite_all',all(math.isfinite(r[k]) for r in gs))
  print('FIRST',gs[0]['timestamp'])
 for p in sorted((d/'training_state').glob('*/complete.json')):
  print('CK',json.loads(p.read_text())['progress'])
 ev=sorted((d/'eval').glob('*'))
 if ev:print('EVAL',ev[-1],len(list(ev[-1].glob('*.png'))))
 for p in d.glob('**/scores.json'):print('SCORES',p,p.read_text()[:800])
 f=Path('logs')/(name+'.log')
 if f.exists():
  lines=f.read_text(errors='replace').splitlines()
  print('ERRORS',[l[:350] for l in lines if 'OutOfMemoryError:' in l or 'RuntimeError:' in l or 'OSError:' in l][-3:])
for p in Path('.').glob('*归档校验.txt'):print('ARCHIVE',p,p.read_text()[-1200:])
PY
