set -eu
work=/tmp/laion-discovery-904.b9IQDB
ls -la "$work/config"
python3 - "$work/config" <<'PY'
import pathlib,json,sys
for p in pathlib.Path(sys.argv[1]).rglob('*'):
 if p.is_file():
  try:
   doc=json.loads(p.read_text())
   print(p.name,list(doc) if isinstance(doc,dict) else type(doc).__name__)
  except Exception:
   print(p.name,p.stat().st_size)
PY
