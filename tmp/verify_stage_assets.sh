set -euo pipefail
root=/data1/shenglin.zeng/teacher_turning_20261026
cd "$root"
python3 -u - <<'PY'
import hashlib,json
from pathlib import Path
p=Path('stage_assets');m=json.loads((p/'manifest.json').read_text())
for name,entry in m.items():
 h=hashlib.sha256()
 with (p/name).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 assert h.hexdigest()==entry['sha256'] and (p/name).stat().st_size==entry['bytes'],name
 print('VERIFIED',name,flush=True)
Path('stage_assets/verified.json').write_text(json.dumps({'passed':True,'files':len(m)}))
PY
