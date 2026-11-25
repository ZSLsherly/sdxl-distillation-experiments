cd /data5/shenglin.zeng/a50k_20261124
date
tail -5 logs/queue.log
for f in logs/first_update* logs/A50K_G*.log; do if [ -f "$f" ]; then echo "$f"; tail -4 "$f" | cut -c1-250; grep -E 'OutOfMemoryError|RESUMED' "$f" | tail -1 |cut -c1-350; fi;done
nvidia-smi --query-gpu=index,memory.total,memory.used,utilization.gpu --format=csv,noheader
df -h /data5
python3 - <<'PY'
import json,statistics
from pathlib import Path
for d in sorted(Path('outputs').iterdir()):
 p=d/'status.json'
 if p.exists():print(d.name, p.read_text()[:1100])
 rows=[]
 for line in (d/'rank0.jsonl').read_text().splitlines() if (d/'rank0.jsonl').exists() else []:
  try:rows.append(json.loads(line))
  except ValueError: pass
 g=[r for r in rows if r.get('phase')=='student_done'];print('n_g',len(g),'recent',[(r['g'],round(r['cycle_seconds'],2),round(r['seconds'],2),round(r['peak_allocated_GiB'],2)) for r in g[-8:]])
 print('last_phase',[(r.get('phase'),r.get('g'),r.get('f'),r.get('seconds')) for r in rows[-3:]])
PY
