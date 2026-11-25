set -eu
. /root/sdxl_research/env.sh
python - <<'PY'
from pathlib import Path
import hashlib,os
o=Path(os.environ['SDXL_WORK'])/'outputs'
for f in sorted(o.rglob('*')):
 if f.is_file() and not f.is_symlink() and f.suffix in ('.pt','.pth','.safetensors','.ckpt','.bin') and f.stat().st_size>1024**2:
  with f.open('rb') as fd:h=hashlib.file_digest(fd,'sha256').hexdigest()
  print(f.relative_to(o),f.stat().st_size,h,flush=True)
PY
nvidia-smi --query-compute-apps=pid --format=csv,noheader
df -h "$SDXL_WORK"
