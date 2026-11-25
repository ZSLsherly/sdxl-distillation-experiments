set -eu
. /root/sdxl_research/env.sh
python - <<'PY'
from pathlib import Path
import os,hashlib
o=Path('/root/autodl-tmp/sdxl_runs/outputs/turning_BC_paired_quota_20261102').resolve(strict=True)
files={
'uniform_control/weights/20261102T203507+0800_g000160_end/student_rank0.pt':'00e8abc9e66f5ebeca6e4733e3b981c0fcc34efe3583100ac565d6a8def8e695',
'B/weights/20261102T065317+0800_g000160_end/student_rank0.pt':'78045ea389b6c781ae81b2b096756ab8104dfe02911cd7cfc920bfc3e9d1aa46',
'C/weights/20261102T134325+0800_g000160_end/student_rank0.pt':'9b117072217a8825ba7a2978f244e3cbfa251aff6b9267c49d9cae18e4c3294b'}
for name,expected in files.items():
 f=o/name
 assert f.resolve(strict=True)==o/name and not f.is_symlink()
 with f.open('rb') as r:
  before=os.fstat(r.fileno())
  assert before.st_size==5135506134 and hashlib.file_digest(r,'sha256').hexdigest()==expected
  now=f.stat()
  assert now.st_ino==before.st_ino and now.st_mtime_ns==before.st_mtime_ns
  f.unlink()
 print('REMOVED_VERIFIED_SOURCE',name,flush=True)
PY
df -B1 --output=avail,target "$SDXL_WORK"
