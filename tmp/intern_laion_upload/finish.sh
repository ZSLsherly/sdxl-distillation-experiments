set -eu
python3 - <<'PY'
from pathlib import Path
p=Path('/tmp/laion-discovery-904.b9IQDB/config/credentials')
assert p.is_file()
p.unlink()
print('temporary CLI credential removed',not p.exists())
PY
