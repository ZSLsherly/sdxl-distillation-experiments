python3 - <<'PY'
import json,time
from pathlib import Path
p=Path('/data1/shenglin.zeng/dmd2_matched_10k_20261119/runs/formal_g160/status.json')
s=json.loads(p.read_text());assert (s['G'],s['F'])==(54,268)
s.update(status='interrupted',reason='guard detected external GPU PID1180063; own process terminated',verified_at=time.time(),resumable=False)
t=p.with_suffix('.tmp');t.write_text(json.dumps(s,indent=2));t.replace(p)
print(json.dumps(s))
PY
