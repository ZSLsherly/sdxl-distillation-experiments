set -euo pipefail
source /data5/shenglin.zeng/env.sh
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
root=/data5/shenglin.zeng/sdxl_crossdistill/outputs/turning_boundary_ablation_20261027
cd "$root/code/sdxl_research"
python smoke_turning_boundary_cpu.py
cp "$root/code/output/sdxl-crossdistill/turning_ablation_20261027/boundary_cpu_smoke.json" "$root/cpu_smoke.json"
python - <<'PY'
import json,hashlib
from pathlib import Path
code=Path.cwd();root=code.parent.parent
d=json.loads((root/'deployment.json').read_text())
d['server_CPU_validation']='passed both arms, each 4G + 8fake, no GAN, no GPU'
d['source_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in code.glob('*.py')}
d['status']='ready_waiting_GPU_resources'
(root/'deployment.json').write_text(json.dumps(d,indent=2))
print('CPU_VALIDATED_NO_GPU_STARTED')
PY
