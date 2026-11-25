source /data5/shenglin.zeng/env.sh
cd /data5/shenglin.zeng/sdxl_crossdistill
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 python -B - <<'PY'
import json, torch
from pathlib import Path
from diffusers import AutoencoderKL
torch.set_num_threads(2)
d=Path('datasets/parallel_10k_seed20261018')
m=json.loads((d/'manifest.json').read_text())
x=torch.load(d/'prepared.pt',mmap=True,weights_only=True)
assert x['train_count']==10000 and len(x['latents'])==10032
assert not set(m['train_original_indices']) & set(m['holdout_original_indices'])
print('DATA_VERIFIED', x['latents'].shape, x['prompt_embeds'].shape)
v=AutoencoderKL(block_out_channels=(8,16), down_block_types=('DownEncoderBlock2D',)*2, up_block_types=('UpDecoderBlock2D',)*2, norm_num_groups=4, sample_size=16).eval().requires_grad_(False)
z=torch.randn(2,4,8,8,requires_grad=True)
y=v.decode(z).sample; g=torch.autograd.grad(y.square().mean(),z)[0]
v.enable_gradient_checkpointing()
z2=z.detach().clone().requires_grad_(True)
y2=v.decode(z2).sample; g2=torch.autograd.grad(y2.square().mean(),z2)[0]
torch.testing.assert_close(y,y2);torch.testing.assert_close(g,g2)
print('VAE_BLOCK_CHECKPOINT_VALUES_AND_GRADIENTS_MATCH')
PY
