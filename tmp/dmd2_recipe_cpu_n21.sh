set -eu
export CUDA_VISIBLE_DEVICES=""
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
R=/data1/shenglin.zeng/dmd2_official_recipe_20261121
cd "$R/code"
"$R/env/bin/python" test_recipe.py
"$R/env/bin/python" test_cpu.py
"$R/env/bin/python" - <<'PY'
import json,hashlib
from pathlib import Path
r=Path('..'); checks={str(p.relative_to(r)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (r/'code').rglob('*.py')}
checks['config.json']=hashlib.sha256((r/'config.json').read_bytes()).hexdigest()
(r/'source_manifest.json').write_text(json.dumps(checks,indent=2))
print('CPU_VALIDATION_AND_MANIFEST_READY',len(checks))
PY
