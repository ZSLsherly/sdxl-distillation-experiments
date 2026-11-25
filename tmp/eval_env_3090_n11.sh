set -eu
r=/data1/shenglin.zeng/teacher_turning_20261026
"$r/env/bin/python" - <<'PY'
import torch,diffusers,importlib.util
print(torch.__version__,diffusers.__version__)
for n in ['cleanfid','scipy','numpy','PIL','transformers']:
 print(n,bool(importlib.util.find_spec(n)))
PY
ls -lh "$r/models/sdxl-base-1.0" "$r/models/sdxl-base-1.0/text_encoder_2" | head -22
