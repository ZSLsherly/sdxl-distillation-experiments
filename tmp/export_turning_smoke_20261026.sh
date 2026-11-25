set -euo pipefail
root=/data1/shenglin.zeng/teacher_turning_20261026
cd "$root"
test -f smoke_gpu0/monitor.json
test -f smoke_gpu0/worker0_complete.json
python3 - <<'PY'
import json,zipfile
from pathlib import Path
p=Path('smoke_gpu0')
for name in ['monitor.json','worker0_complete.json']:
 d=json.loads((p/name).read_text())
 if name=='monitor.json':
  print('MONITOR',d['returncode'],d['stop_reason'],'wall_seconds',d['wall_seconds'],'min_free_MiB',min(x['free_MiB'] for x in d['samples']))
 else:print('COMPLETE',d)
d=json.loads((p/'base_dt10_bf16_prompt000_seed201.json').read_text())
print('TRAJECTORY',{k:d[k] for k in ['seconds','peak_allocated_GiB','prompt','dtype']},'intervals',len(d['rows']))
with zipfile.ZipFile('smoke_gpu0.zip','w',zipfile.ZIP_DEFLATED) as z:
 for f in p.iterdir():
  if f.suffix in ['.json','.png','.log']:z.write(str(f),f.name)
PY
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits -i 0
