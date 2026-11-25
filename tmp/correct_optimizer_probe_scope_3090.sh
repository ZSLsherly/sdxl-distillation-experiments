set -eu
python3 - <<'PY'
from pathlib import Path
import json
p=Path('/data1/shenglin.zeng/teacher_turning_20261026/outputs/turning_boundary_ablation_20261027/optimizer_probe.json')
if p.exists():
 d=json.loads(p.read_text());d['scope']='Initial small GPU DTensor diagnostic: from_local moved input tensors to CUDA. Excluded from CPU-offload conclusions.'
 d['superseded_for_CPU_analysis_by']='optimizer_probe_cpu.json, which asserts parameter/gradient/moment devices are CPU'
 p.write_text(json.dumps(d,indent=2))
print('DIAGNOSTIC_SCOPE_CORRECTED')
PY
nvidia-smi -i 0 --query-compute-apps=pid,used_memory --format=csv,noheader,nounits
