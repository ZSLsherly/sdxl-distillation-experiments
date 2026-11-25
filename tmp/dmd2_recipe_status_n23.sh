R=/data1/shenglin.zeng/dmd2_official_recipe_20261121
date
tail -n 18 "$R/logs/queue.log" 2>/dev/null || true
ps -eo pid,etime,args | grep -E '[t]rain_matched.py|[r]un_guard.py|[e]valuate_dmd2.py|[r]un_queue.py' || true
"$R/env/bin/python" - <<'PY'
from pathlib import Path
import json,math
r=Path('/data1/shenglin.zeng/dmd2_official_recipe_20261121')
for p in [r/'smoke_passed.json',r/'complete.json',*sorted((r/'runs').glob('*/status.json')),r/'coco10k/complete.json',r/'coco10k/comparison.json',*sorted((r/'coco10k').glob('status_*.json'))]:
 if p.exists():print(str(p.relative_to(r)),p.read_text())
p=r/'runs/formal_g160/rank0.jsonl'
if p.exists():
 rows=[json.loads(x) for x in p.read_text().splitlines() if x.strip()];g=[x for x in rows if x.get('student_updated')]
 print('ROWS',len(rows),'G_ROWS',len(g));print('LAST_UPDATES',json.dumps(rows[-2:]));print('LAST_G',json.dumps(g[-1:]));print('ALL_LOG_NUMBERS_FINITE',all(not isinstance(v,float) or math.isfinite(v) for x in rows for v in x.values()))
 print('G160_IMAGES',len(list((r/'runs/formal_g160/eval/g000160').glob('*.png'))))
 ck=r/'runs/formal_g160/student.pt';print('FINAL_WEIGHT_BYTES',ck.stat().st_size if ck.exists() else 0)
PY
tail -n 8 "$R/coco10k/logs/queue.log" 2>/dev/null || true
df -h /data1
