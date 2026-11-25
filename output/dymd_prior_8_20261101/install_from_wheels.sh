#!/bin/bash
set -euo pipefail
B=/data1/shenglin.zeng/dymd_prior_8_20261101
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple/
"$B/env/bin/pip" install "$B"/wheels/torch-*.whl "$B"/wheels/torchvision-*.whl "$B"/wheels/torchaudio-*.whl
"$B/env/bin/pip" install "$B"/wheels/flash_attn-*.whl
"$B/env/bin/pip" install -r "$B/official/pf_wan/requirements.txt" wandb pyarrow pandas matplotlib
cd "$B/official/pf_wan"
PYTHONPATH="$PWD" "$B/env/bin/python" - <<'PY'
import torch,flash_attn
from project.engines import ENGINE_REGISTRY
from project.models.meta_model.wan2_1_i2v import Wan2_1_I2V
print('IMPORT_OK',torch.__version__,torch.version.cuda,flash_attn.__version__)
PY
"$B/env/bin/python" "$B/test_measurement.py"
"$B/env/bin/pip" freeze > "$B/environment.txt"
touch "$B/ENV_READY"
