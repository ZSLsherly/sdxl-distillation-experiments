set -eu
BASE=/data1/shenglin.zeng/dymd_prior_8_20261101
export http_proxy=http://127.0.0.1:18897 https_proxy=http://127.0.0.1:18897
export HF_HUB_DISABLE_XET=1 HF_HUB_DOWNLOAD_TIMEOUT=180
/data1/shenglin.zeng/miniconda3/envs/sherly/bin/python - <<'PY'
from huggingface_hub import snapshot_download
import json,pathlib
b=pathlib.Path('/data1/shenglin.zeng/dymd_prior_8_20261101')
sha=json.loads((b/'model_api.json').read_text())['sha']
p=snapshot_download('DAGroup-PKU/PF_Wan',revision=sha,local_dir=str(b/'official/pf_wan/checkpoints/PF_Wan'),max_workers=4)
print('DOWNLOAD_COMPLETE',p,flush=True)
(b/'MODEL_READY').touch()
PY
