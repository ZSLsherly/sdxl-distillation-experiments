set -euo pipefail
root=/data1/shenglin.zeng/teacher_turning_20261026
cd "$root"
export LD_LIBRARY_PATH="$root/runtime/lib:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES=''
env/bin/python -u - <<'PY'
import hashlib,json,torch
from pathlib import Path
p=Path('datasets/parallel_10k_seed20261018');checks=json.loads((p/'transfer_checksums.json').read_text())
for name,m in checks.items():
 h=hashlib.sha256()
 with (p/name).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 assert h.hexdigest()==m['sha256'] and (p/name).stat().st_size==m['bytes']
 print('VERIFIED',name,flush=True)
d=torch.load(p/'prepared.pt',weights_only=True,mmap=True,map_location='cpu');assert d['train_count']==10000
print('TRAIN_COUNT',d['train_count'],'LATENTS',list(d['latents'].shape),flush=True)
Path('outputs/turning_boundary_ablation_20261027/data_verified.json').write_text(json.dumps(checks,indent=2))
PY
