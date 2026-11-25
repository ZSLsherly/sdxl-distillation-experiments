set -eu
r=/data1/shenglin.zeng/coco_missing_20261111
p=/data1/shenglin.zeng/teacher_turning_20261026/env/bin/python
cd "$r"
tar -xzf code_bundle.tgz
"$p" - <<'PY'
import sys
from pathlib import Path
print(sys.version)
for f in Path('code').glob('*.py'):compile(f.read_text(),str(f),'exec')
import torch, numpy, diffusers, cleanfid
print('EVALUATOR_IMPORTS_OK',torch.__version__,diffusers.__version__)
PY
