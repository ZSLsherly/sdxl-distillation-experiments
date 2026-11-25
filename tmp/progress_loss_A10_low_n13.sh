set -eu
. /root/sdxl_research/env.sh
python - <<'PY'
from pathlib import Path
from datetime import datetime,timezone,timedelta
import json,math,statistics
r=Path('/root/sdxl_research/outputs/A_pcm10_terminal_low30_g160_20261113');now=datetime.now(timezone(timedelta(hours=8)))
print('NOW',now.isoformat())
print('STATUS',json.loads((r/'model/status.json').read_text()))
rows=[]
for l in (r/'model/rank0.jsonl').read_text().splitlines():
 try:rows.append(json.loads(l))
 except json.JSONDecodeError:pass
G=[x for x in rows if x.get('phase')=='student_done'];F=[x for x in rows if x.get('phase')=='fake']
print('COUNTS',len(G),len(F));print('LAST_G',G[-1] if G else None)
for name,vals,keys in [('student',G,['pcm_loss','dmd_loss','middle_dm_loss','terminal_dm_loss','grad_norm','cycle_seconds']),('fake',F,['fake_loss','fake_grad_norm'])]:
 for label,v in [('first10',vals[:10]),('previous10',vals[-20:-10]),('last10',vals[-10:])]:
  print(name,label,{k:{'mean':statistics.mean(x[k] for x in v),'min':min(x[k] for x in v),'max':max(x[k] for x in v)} for k in keys if v and all(k in x for x in v)})
print('FAKE_PER_STAGE',{str(k):{'n':len(v),'first10_mean':statistics.mean(x['fake_loss'] for x in v[:10]),'last10_mean':statistics.mean(x['fake_loss'] for x in v[-10:]),'last10_grad':statistics.mean(x.get('fake_grad_norm',0) for x in v[-10:])} for k in [2,3,4] if (v:=[x for x in F if x.get('fake_stage')==k])})
bad=[]
for x in G+F:
 for k,v in x.items():
  if isinstance(v,float) and not math.isfinite(v):bad.append((x.get('g',x.get('fake_update')),k))
print('NONFINITE',bad[:15]);print('CLIP_COUNTS',{'student_gt1':sum(x.get('grad_norm',0)>1 for x in G),'fake_gt1':sum(x.get('fake_grad_norm',0)>1 for x in F)})
if len(G)>=2:
 n=min(10,len(G)-1);window=G[-n:];cycles=[x['cycle_seconds'] for x in window]
 def dt(x):return datetime.strptime(x['timestamp'],'%Y%m%dT%H%M%S%z')
 wall=(dt(G[-1])-dt(G[-n-1])).total_seconds()/n
 remaining=160-G[-1]['g'];print('ETA',{'remaining_G':remaining,'recent_n':n,'cycle_mean_seconds':statistics.mean(cycles),'wall_seconds_per_G':wall,'remaining_hours':remaining*wall/3600,'last_complete':G[-1]['timestamp']})
print('EVAL_LAST',sorted(str(p) for p in (r/'model/eval').glob('*/rank0.json'))[-2:])
print('QUEUE_TAIL',(r/'queue.log').read_text()[-700:]);print('TRAIN_TAIL',(r/'train.log').read_text()[-900:])
(r/'progress_snapshot.json').write_text(json.dumps({'now':now.isoformat(),'status':json.loads((r/'model/status.json').read_text()),'student':G,'fake':F},ensure_ascii=False))
PY
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
df -h /root/autodl-tmp
ps -eo pid,ppid,comm,args | awk '$3 ~ /python|torchrun/ && /A_pcm10_terminal_low30|run_A_terminal_prior/ {print substr($0,1,200)}'
