cd /data5/shenglin.zeng/a50k_20261124 || exit 1
date
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader
tail -4 logs/singlepass_queue.log
tail -5 logs/A50K_singlepass.log | cut -c1-240
grep -E 'OutOfMemory|RuntimeError:|Error:' logs/A50K_singlepass.log | tail -3 | cut -c1-450
python3 - <<'PY'
from pathlib import Path
import json,statistics,datetime
d=Path('outputs/A50K_singlepass')
rows=[]
for line in (d/'rank0.jsonl').read_text().splitlines():
 try:rows.append(json.loads(line))
 except ValueError:pass
gs=[r for r in rows if r['phase']=='student_done'];fs=[r for r in rows if r['phase']=='fake']
print('STATUS',(d/'status.json').read_text()[:1600])
print('FAKE', [{k:v for k,v in r.items() if k in ('phase','fake_update','fake_stage','fake_loss','timestamp')} for r in fs[-12:]])
for r in gs[-6:]:print('G', {k:v for k,v in r.items() if k in ('g','stage','timestamp','pcm_loss','middle_dm_loss','terminal_dm_loss','grad_norm','seconds','cycle_seconds','peak_allocated_GiB','peak_reserved_GiB')})
def dt(r):return datetime.datetime.strptime(r['timestamp'],'%Y%m%dT%H%M%S%z')
for n in [10,30,100]:
 if len(gs)<=n:continue
 a=gs[-n-1];b=gs[-1];wall=(dt(b)-dt(a)).total_seconds()/n
 print('MEAN',n,'wall',wall,'cycle',statistics.mean(r['cycle_seconds'] for r in gs[-n:]),'student',statistics.mean(r['seconds'] for r in gs[-n:]))
print('CHECKPOINTS',[p.parent.name for p in (d/'training_state').glob('*/complete.json')])
PY
