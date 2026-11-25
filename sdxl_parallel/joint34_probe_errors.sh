source /data5/shenglin.zeng/env.sh
cd /data5/shenglin.zeng/sdxl_crossdistill
python - <<'PY'
import json
from pathlib import Path
m=json.loads(Path('outputs/gan_stage_joint34/probe_launch.json').read_text())
p=Path(m['exit']);print('EXIT',p.read_text() if p.exists() else 'pending')
lines=Path(m['log']).read_text(errors='replace').splitlines()
for i,s in enumerate(lines):
 if any(x in s for x in ['Error','ERROR','Exception','WARN','terminated','Traceback','Root Cause','remote process','Connection']):
  print('\n'.join(lines[max(0,i-2):min(len(lines),i+4)]))
print('LAST_UPDATES')
for r in [json.loads(s) for s in (Path('outputs')/m['run']/'rank0.jsonl').read_text().splitlines()]:
 if r['phase']=='student_done':print(json.dumps(r))
PY
