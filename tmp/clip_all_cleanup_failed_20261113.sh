python3 - <<'PY'
from pathlib import Path
import os,shutil
root=Path('/data1/shenglin.zeng/clip_missing_20261113')
assert root.resolve()==root and root.name=='clip_missing_20261113'
assert (root/'selection.json').exists() and (root/'validation.json').exists()
for cmd in Path('/proc').glob('[0-9]*/cmdline'):
 try:args=cmd.read_bytes().split(b'\0')
 except OSError:continue
 assert not any(a.startswith(str(root/'code').encode()) for a in args),'Own queue still active'
# These are copies created by this supplement today, not original archives.
arms=root/'arms';assert arms.resolve()==arms
size=sum(p.stat().st_size for p in arms.rglob('*') if p.is_file())
shutil.rmtree(arms)
(root/'ABANDONED.txt').write_text('CUDA initialization failed. Evaluation moved to 172.18.32.81 /data3/shenglin.zeng/clip_missing_20261113 GPU5. Temporary model/image copies removed; original archives unchanged.\n')
print('REMOVED_OWN_TEMP_GiB',size/2**30)
PY
