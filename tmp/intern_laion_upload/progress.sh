work=/tmp/laion-discovery-904.b9IQDB
python3 - <<'PY'
from pathlib import Path
import re
p=Path('/tmp/laion-discovery-904.b9IQDB/upload.log')
with p.open('rb') as f:
 f.seek(max(0,p.stat().st_size-3500)); tail=f.read().decode(errors='replace')
print('log bytes',p.stat().st_size)
for line in re.split(r'[\r\n]+',tail)[-7:]:
 if not any(t in line.lower() for t in ('http','token','signature','secret','credential')): print(line[:200])
PY
