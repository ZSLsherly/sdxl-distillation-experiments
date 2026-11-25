set -eu
. /root/sdxl_research/env.sh
d=/root/autodl-tmp/h20_gan_mb16
c="$d/code_fp32sam4";mkdir -p "$c";cp "$d/code_fast"/*.py "$d/code_fast/config.json" "$c/"
cd "$c"
python - <<'PY'
from pathlib import Path
p=Path('sam_pixel.py');s=p.read_text()
assert 'self.backbone = backbone.eval().requires_grad_(False).to(torch.bfloat16)' in s
s=s.replace('self.backbone = backbone.eval().requires_grad_(False).to(torch.bfloat16)','self.backbone = backbone.eval().requires_grad_(False)')
a=s.index('        # Native full-image');b=s.index('    def forward(',a)
s=s[:a]+'''        # Chunk only the frozen encoder's image dimension. Heads, R1 and hinge see all 16.
        groups=[]
        for part in x.split(4):
            h=self.backbone.patch_embed(part)
            if self.backbone.pos_embed is not None:h=h+self.backbone.pos_embed
            taps=[]
            for i,layer in enumerate(self.backbone.layers,1):
                h=checkpoint(layer,h,use_reentrant=False) if h.requires_grad else layer(h)
                if i in (3,6,9,12):taps.append(h.permute(0,3,1,2).contiguous())
            groups.append(taps)
        return tuple(torch.cat(taps,dim=0) for taps in zip(*groups))

'''+s[b:];p.write_text(s)
p=Path('run_h20_16.py');s=p.read_text();assert 'h20_gan_mb16_fast' in s;p.write_text(s.replace('h20_gan_mb16_fast','h20_gan_mb16_fp32sam4'))
PY
export PYTHONPATH="$c" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONFAULTHANDLER=1
L=/root/autodl-tmp/h20_gan_mb32/blas126/nvidia/cublas/lib;export LD_PRELOAD="$L/libcublasLt.so.12:$L/libcublas.so.12"
CUDA_VISIBLE_DEVICES='' python - <<'PY'
import torch,importlib.util,copy
from transformers import SamVisionConfig
from transformers.models.sam.modeling_sam import SamVisionEncoder
from sam_pixel import SAMDiscriminator
torch.manual_seed(20261103);torch.set_num_threads(2)
spec=importlib.util.spec_from_file_location('native_sam','../code/sam_pixel.py')
refmod=importlib.util.module_from_spec(spec);spec.loader.exec_module(refmod)
cfg=SamVisionConfig(hidden_size=32,num_hidden_layers=12,num_attention_heads=4,output_channels=16,image_size=64,patch_size=8,window_size=4,global_attn_indexes=[2,5,8,11],mlp_dim=64)
cfg._attn_implementation='sdpa'
old=refmod.SAMDiscriminator(backbone=SamVisionEncoder(cfg));new=SAMDiscriminator(backbone=copy.deepcopy(old.backbone));new.load_state_dict(old.state_dict(),strict=True)
sizes=[];new.backbone.patch_embed.register_forward_pre_hook(lambda m,inp:sizes.append(len(inp[0])))
x=torch.randn(16,3,64,64,requires_grad=True); y=x.detach().clone().requires_grad_()
for m in (old,new):m.set_heads(False)
u=old(x);v=new(y); assert sizes==[4]*4 and len(v)==16
g=torch.autograd.grad(-u.mean(),x)[0];j=torch.autograd.grad(-v.mean(),y)[0]
torch.testing.assert_close(u,v,atol=2e-5,rtol=1e-4);torch.testing.assert_close(g,j,atol=2e-7,rtol=1e-4)
for m in (old,new):m.set_heads(True)
a=old(x.detach(),return_parts=True,feature_r1=True);b=new(x.detach(),return_parts=True,feature_r1=True)
torch.testing.assert_close(a['r1'],b['r1'],atol=2e-5,rtol=1e-4)
ga=torch.autograd.grad(torch.relu(1-a['total']).mean()+1e-5*a['r1'],old.trainable())
gb=torch.autograd.grad(torch.relu(1-b['total']).mean()+1e-5*b['r1'],new.trainable())
for a,b in zip(ga,gb):torch.testing.assert_close(a,b,atol=2e-5,rtol=2e-4)
print('CPU PASS: 4+4+4+4, one 16-image loss; logits/R1/input and head gradients. Input grad max abs diff',float((g-j).abs().max()),flush=True)
PY
nohup python -u run_h20_16.py </dev/null > h20.log 2>&1 & echo JOB_PID=$!
