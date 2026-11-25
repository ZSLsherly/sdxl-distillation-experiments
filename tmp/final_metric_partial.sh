source /data5/shenglin.zeng/env.sh
cd /data5/shenglin.zeng/sdxl_crossdistill
python - <<'PY'
import json,statistics
from pathlib import Path
p=Path('outputs/benchmark_final_20261020/per_image.jsonl')
rs=[]
for line in p.read_text().splitlines():
 try:r=json.loads(line)
 except json.JSONDecodeError:continue
 if r['student_updates']==160:rs.append(r)
for v in ['joint4','gan4']:
 rr=[r for r in rs if r['variant']==v]
 if len(rr)==4:print(v,{m:statistics.mean(r[m] for r in rr) for m in ['niqe','clip_cosine']})
PY
