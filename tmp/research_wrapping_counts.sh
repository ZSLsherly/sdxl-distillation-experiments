source /data5/shenglin.zeng/env.sh
export CUDA_VISIBLE_DEVICES=''
cd /data5/shenglin.zeng/sdxl_crossdistill
python - <<'PY'
import torch,json
from diffusers import UNet2DConditionModel
from diffusers.models.attention import BasicTransformerBlock
from diffusers.models.resnet import Downsample2D,ResnetBlock2D,Upsample2D
from diffusers.models.transformers.transformer_2d import Transformer2DModel
cfg=json.load(open('models/sdxl-base-1.0/unet/config.json'))
with torch.device('meta'): model=UNet2DConditionModel.from_config(cfg)
for mode in ['fine','block']:
 types=(ResnetBlock2D,Transformer2DModel,Downsample2D,Upsample2D)
 if mode=='fine': types+=(BasicTransformerBlock,)
 seen=set();sizes=[]
 for child in list(reversed(list(model.modules())))+[model]:
  if not isinstance(child,types) and child is not model:continue
  ps=[p for p in child.parameters() if id(p) not in seen]
  seen.update(id(p) for p in ps)
  sizes.append(sum(p.numel() for p in ps))
 print(json.dumps({'mode':mode,'groups':len(sizes),'nonempty_groups':sum(n>0 for n in sizes),'total_params':sum(sizes),'largest_group_BF16_MiB':max(sizes)*2/2**20,'five_four_step_rollouts_group_visits':sum(n>0 for n in sizes)*20}))
PY
