set -eu
. /root/sdxl_research/env.sh
base=/root/autodl-tmp/h20_gan_mb16
cd "$base/code_fp32sam4"
L=/root/autodl-tmp/h20_gan_mb32/blas126/nvidia/cublas/lib;export LD_PRELOAD="$L/libcublasLt.so.12:$L/libcublas.so.12"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
test -f "$SDXL_ROOT/outputs/h20_gan_mb16_fp32sam4/benchmark.json"
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)"
python - <<'PY'
from pathlib import Path
import os,importlib.util,json
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL
from sam_pixel import SAMDiscriminator
from image_views import decode_view
r=Path(os.environ['SDXL_ROOT']);d=r/'datasets/parallel_10k_seed20261018'
v=torch.load(d/'prepared.pt',weights_only=True,mmap=True,map_location='cpu')
m=json.loads((d/'manifest.json').read_text());ids=m['eval_original_indices'];idx=[v['original_indices'].index(i) for i in ids]
assert len(idx)==4 and min(idx)>=v['train_count']
vae=AutoencoderKL.from_pretrained(r/'models/sdxl-base-1.0',subfolder='vae',variant='fp16',torch_dtype=torch.float32,local_files_only=True).eval().to('cuda')
with torch.no_grad():rgb=decode_view(vae,v['latents'][idx].float().cuda(),None,kind='whole',chunk_size=1)
del vae,v
spec=importlib.util.spec_from_file_location('sam_bf16','../code_fast/sam_pixel.py')
bf=importlib.util.module_from_spec(spec);spec.loader.exec_module(bf)
sam=Path(os.environ['SDXL_WORK'])/'flash_gan_inputs/sam-vit-base'
torch.manual_seed(20261017+7000)
fp=SAMDiscriminator(sam).eval().cuda(); b=bf.SAMDiscriminator(sam).eval().cuda()
b.load_state_dict(fp.state_dict(),strict=True)
def probe(mod):
    mod.set_heads(False)
    x=rgb.detach().requires_grad_()
    taps=mod.features(x)
    scores=torch.stack([head(t).mean((1,2,3)) for head,t in zip(mod.heads,taps)]).mean(0)
    grad=torch.autograd.grad(-scores.mean(),x)[0]
    return [t.detach().flatten(1).cpu() for t in taps],scores.detach().cpu(),grad.detach().flatten(1).cpu()
a,sa,ga=probe(fp);q,sq,gq=probe(b)
def comp(x,y):
    x=x.double();y=y.double()
    return {'cosine_each':F.cosine_similarity(x,y,dim=1).tolist(),'relative_l2_each':((x-y).norm(dim=1)/x.norm(dim=1)).tolist()}
res=dict(ids=ids,real_vae_decoded=True,head_state='same randomly initialized FP32 weights in both; gradient at pixels, no trained GAN quality inference',features=[comp(x,y) for x,y in zip(a,q)],pixel_gradient=comp(ga,gq),fp32_scores=sa.tolist(),bf16_scores=sq.tolist())
Path('sam_precision_probe.json').write_text(json.dumps(res,indent=2))
print('taps min cosine',min(v for t in res['features'] for v in t['cosine_each']),'gradient cosine',res['pixel_gradient']['cosine_each'],'relative gradient L2',res['pixel_gradient']['relative_l2_each'])
PY
