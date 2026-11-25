set -eu
R=/data1/shenglin.zeng/dmd2_official_recipe_4gpu_20261123
printf '%s\n' 'User requested microbatch16 probe before formal training; stop only this queue.' > "$R/STOP_REQUEST"
sleep 8
"$R/env/bin/python" - <<'PY'
from pathlib import Path
import json
r=Path('/data1/shenglin.zeng/dmd2_official_recipe_4gpu_20261123').resolve()
for proc in Path('/proc').iterdir():
 if not proc.name.isdigit():continue
 try:args=(proc/'cmdline').read_bytes().decode(errors='replace').split('\0')
 except OSError:continue
 assert not any(str(r/'code'/f) in args for f in ['train_matched.py','run_guard.py']),'Own previous process still running'
p=r/'runs/formal_g160'
if p.exists():
 rows=[json.loads(x) for x in (p/'rank0.jsonl').read_text().splitlines()] if (p/'rank0.jsonl').exists() else []
 dest=r/'runs/formal_superseded_for_micro16'
 assert not dest.exists() and p.resolve().is_relative_to(r)
 (p/'status.json').write_text(json.dumps({'status':'superseded_for_user_micro16_test','G':rows[-1]['G'] if rows else 0,'F':rows[-1]['F'] if rows else 0},indent=2))
 p.rename(dest)
 print('ARCHIVED_EARLY_FORMAL',len(rows),'updates')
(r/'STOP_REQUEST').unlink()
print('OWN_QUEUE_STOPPED_READY_FOR_MICRO16')
PY
