R=/data1/shenglin.zeng/dmd2_official_recipe_20261121
tail -n 110 "$R/logs/formal_g160.log"
find "$R/runs/formal_g160" -maxdepth 2 -type f \( -name '*.pt' -o -name '*.safetensors' -o -name '*complete*' \) -printf '%p %s bytes\n'
nvidia-smi --query-gpu=index,uuid,memory.free,utilization.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader
"$R/env/bin/python" - <<'PY'
from pathlib import Path
import json,datetime,statistics
r=Path('/data1/shenglin.zeng/dmd2_official_recipe_20261121/runs/formal_g160')
rows=[json.loads(x) for x in (r/'rank0.jsonl').read_text().splitlines()];g=[x for x in rows if x['student_updated']]
tz=datetime.timezone(datetime.timedelta(hours=8))
print('LAST_UPDATE_BEIJING',datetime.datetime.fromtimestamp(rows[-1]['time'],tz).isoformat())
for lo,hi in [(1,20),(81,100),(101,106)]:
 a=[x for x in g if lo<=x['G']<=hi];b=[x for x in rows if lo<=x['G']<=hi]
 print('WINDOW',lo,hi,json.dumps({k:statistics.mean(x[k] for x in (a if k.startswith('student') else b)) for k in ['student_dm','student_gan','student_grad','fake_mse','fake_gan','fake_grad']}))
print('EVAL_TAIL',[(x.name,len(list(x.glob('*.png')))) for x in sorted((r/'eval').glob('g*'))[-5:]])
PY
