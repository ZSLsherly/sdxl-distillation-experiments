set -eu
r=/data1/shenglin.zeng/dmd2_matched_10k_20261119
mkdir -p "$r"
free -h
base=/data1/shenglin.zeng/sdxl_eval_uniform_20261025
CUDA_VISIBLE_DEVICES='' "$base/env/bin/python" - <<'PY'
from pathlib import Path
import torch,json,hashlib
p=Path('/data1/shenglin.zeng/sdxl_eval_uniform_20261025/datasets/parallel_10k_seed20261018')
d=torch.load(p/'prepared.pt',weights_only=True,map_location='cpu',mmap=True)
print('CACHE', {k:list(v.shape) if torch.is_tensor(v) else (v if isinstance(v,(int,str,float)) else type(v).__name__) for k,v in d.items()})
print('TIME_IDS',d['time_ids'][:1].tolist())
print('NEGATIVE_MAX',d['negative_prompt_embeds'].abs().max().item(),d['negative_pooled_prompt_embeds'].abs().max().item())
print('MANIFEST_SHA256',hashlib.sha256((p/'manifest.json').read_bytes()).hexdigest())
PY
test -d "$r/env" || "$base/env/bin/python" -m venv --system-site-packages "$r/env"
"$r/env/bin/python" -m pip install 'peft==0.17.1' 'matplotlib==3.10.7' imageio
