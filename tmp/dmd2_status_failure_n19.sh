R=/data1/shenglin.zeng/dmd2_matched_10k_20261119
ps -eo pid,etime,args | grep -E '[r]un_guard.py|[t]rain_matched.py|1180063' || true
tail -n 18 "$R/logs/queue.log"
tail -n 12 "$R/logs/formal_g160.log"
stat -c '%y %n' "$R/logs/queue.log" "$R/runs/formal_g160/rank0.jsonl"
find "$R/runs/formal_g160" -maxdepth 2 -type f -name '*.pt' -o -name 'complete.json'
"$R/env/bin/python" - <<'PY'
import json,math,statistics
from pathlib import Path
r=Path('/data1/shenglin.zeng/dmd2_matched_10k_20261119/runs/formal_g160')
a=[json.loads(s) for s in (r/'rank0.jsonl').read_text().splitlines()];g=[s for s in a if s['student_updated']]
print(json.dumps({'rows':len(a),'last':a[-1],'first_G':g[0],'last_G':g[-1],'finite':all(math.isfinite(v) for s in a for v in s.values() if isinstance(v,float)),'last50_seconds':sum(s['seconds'] for s in a[-50:]),'eval_dirs':[p.name for p in sorted((r/'eval').iterdir())][-5:]},indent=2))
PY
for i in 0 1 3 5 6 7; do nvidia-smi -i "$i" --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader 2>&1; done
