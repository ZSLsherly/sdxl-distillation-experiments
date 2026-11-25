cd /data5/shenglin.zeng/a50k_20261124 || exit 1
date
nvidia-smi --query-gpu=index,uuid,utilization.gpu,clocks.current.sm,power.draw,power.limit,temperature.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader
python3 - <<'PY'
from pathlib import Path
from datetime import datetime
import json,statistics
def wall(r):return datetime.strptime(r['timestamp'],'%Y%m%dT%H%M%S%z').timestamp()
configs=[]
for name in ('A50K_singlepass','A50K_no_middle_singlepass'):
 d=Path('outputs')/name
 configs.append(json.loads((d/'config.json').read_text()))
 rows=[]
 for l in (d/'rank0.jsonl').read_text().splitlines():
  try:rows.append(json.loads(l))
  except ValueError:pass
 gs={r['g']:r for r in rows if r['phase']=='student_done'};fs={r['fake_update']:r for r in rows if r['phase']=='fake'}
 print(name,'latest',max(gs))
 for beg,end in [(11,50),(101,140),(201,240),(351,390),(501,540),(597,616),(11,616)]:
  if max(gs)<end:continue
  ids=[i for i in range(beg,end+1) if gs[i]['effective_batch']==64]
  def mean(f):return round(statistics.mean(f(i) for i in ids),3)
  print(beg,end,'fake1 fake2 student cycle',mean(lambda i:fs[2*i-1]['seconds']),mean(lambda i:fs[2*i]['seconds']),mean(lambda i:gs[i]['seconds']),mean(lambda i:gs[i]['cycle_seconds']),'wall',round((wall(gs[end])-wall(gs[beg-1]))/(end-beg+1),3))
 print('stages@616',gs[616]['supervised_stages'])
print('diff_cfg',[(k,configs[0].get(k),configs[1].get(k)) for k in configs[0] if configs[0].get(k)!=configs[1].get(k) and k!='experiment_modules'])
PY
