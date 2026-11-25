R=/data1/shenglin.zeng/dmd2_matched_10k_20261119
"$R/env/bin/python" - <<'PY'
from pathlib import Path
import json,math
r=Path('/data1/shenglin.zeng/dmd2_matched_10k_20261119/runs/formal_g160_restart_20261120')
rows=[]
for k in range(4):
 a=[json.loads(x) for x in (r/f'rank{k}.jsonl').read_text().splitlines()]
 assert a and all(math.isfinite(v) for x in a for v in x.values() if isinstance(v,float))
 rows.append({'rank':k,'G':a[-1]['G'],'F':a[-1]['F'],'peak_GiB':a[-1]['peak_GiB']})
print(json.dumps({'ranks':rows,'G0_images':len(list((r/'eval/g000000').glob('*.png'))),'all_finite':True}))
PY
