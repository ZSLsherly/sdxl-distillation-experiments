from pathlib import Path
import os,json
import torch
from diffusers import AutoencoderKL
from image_views import decode_view

root=Path(os.environ['SDXL_ROOT'])
dataset=root/'datasets/parallel_10k_seed20261018'
data=torch.load(dataset/'prepared.pt',mmap=True,weights_only=True,map_location='cpu')
ids=json.loads((dataset/'manifest.json').read_text())['eval_original_indices']
indices=[data['original_indices'].index(i) for i in ids]
z=data['latents'][indices].float().cuda()
vae=AutoencoderKL.from_pretrained(root/'models/sdxl-base-1.0',subfolder='vae',variant='fp16',torch_dtype=torch.float32,local_files_only=True).eval().cuda().requires_grad_(False)
def run(n):
    x=z.detach().requires_grad_()
    rgb=decode_view(vae,x,None,kind='whole',grad=True,chunk_size=n)
    # Same spatially varying linear objective in both runs.
    weight=torch.linspace(-1,1,rgb.shape[-1],device=rgb.device)
    g,=torch.autograd.grad((rgb*weight).mean(),x)
    return rgb.detach().cpu(),g.cpu()
a,g=run(1)
b,h=run(2)
def rel(x,y):return ((x-y).norm()/x.norm().clamp_min(1e-30)).item()
out={'ids':ids,'rgb_relative_l2':rel(a,b),'latent_grad_relative_l2':rel(g,h),'grad_norm':g.norm().item()}
out['pass']=out['grad_norm']>0 and max(out['rgb_relative_l2'],out['latent_grad_relative_l2'])<.001
Path('vae_compare.json').write_text(json.dumps(out,indent=2))
print(out,flush=True)
