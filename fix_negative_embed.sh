set -eu
source /data5/shenglin.zeng/env.sh
python - <<'PY'
from pathlib import Path
import torch
p=Path('/data5/shenglin.zeng/models/negative_embed.pt');x=torch.load(p,weights_only=True)
if x.ndim==3 and x.shape[0]==1:
 torch.save(x[0],p)
 x=x[0]
assert x.shape==(512,4096),x.shape
print('NEGATIVE_EMBED_TRAIN_SHAPE',x.shape)
PY
