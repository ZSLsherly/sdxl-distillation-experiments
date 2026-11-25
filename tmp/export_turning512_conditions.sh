set -euo pipefail
source /data5/shenglin.zeng/env.sh
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
python -u - <<'PY'
import torch,json,hashlib
from pathlib import Path
root=Path('/data5/shenglin.zeng/sdxl_crossdistill')
d=torch.load(root/'datasets/prepared.pt',weights_only=True,map_location='cpu')
ids=[];seen=set()
for i,p in enumerate(d['prompts']):
 if p not in seen:ids.append(i);seen.add(p)
 if len(ids)==256:break
print('CACHE_SIZE',len(d['prompts']),'SELECTED_UNIQUE',len(ids),flush=True)
small={'prompts':[d['prompts'][i] for i in ids],'source_indices':ids}
for k in ('prompt_embeds','negative_prompt_embeds','pooled_prompt_embeds','negative_pooled_prompt_embeds','time_ids'):
 small[k]=d[k] if len(d[k])==1 else d[k][ids]
out=root/'exports/turning_20261026/conditions512.pt';torch.save(small,out)
print('EXPORTED',out.stat().st_size,hashlib.sha256(out.read_bytes()).hexdigest(),flush=True)
PY
