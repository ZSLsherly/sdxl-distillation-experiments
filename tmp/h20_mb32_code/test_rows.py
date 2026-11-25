import copy,torch
from transformers import SamVisionConfig
from transformers.models.sam.modeling_sam import SamVisionEncoder
from sam_rows import install_rows,backbone_features
torch.manual_seed(17);torch.set_num_threads(2)
cfg=SamVisionConfig(hidden_size=32,num_hidden_layers=12,num_attention_heads=4,output_channels=16,image_size=64,patch_size=8,window_size=4,global_attn_indexes=[2,5,8,11],mlp_dim=64)
cfg._attn_implementation='sdpa'
ref=SamVisionEncoder(cfg).eval().requires_grad_(False);alt=copy.deepcopy(ref);install_rows(alt)
x=torch.randn(3,3,64,64,requires_grad=True);y=x.detach().clone().requires_grad_()
h=ref(x,output_hidden_states=True).hidden_states
a=tuple(h[i].permute(0,3,1,2).contiguous() for i in (3,6,9,12));b=backbone_features(alt,y)
vs=[torch.randn_like(t) for t in a]
for aa,bb in zip(a,b):torch.testing.assert_close(aa,bb,rtol=3e-5,atol=1e-5)
g=torch.autograd.grad(sum((u*v).sum() for u,v in zip(a,vs)),x)[0]
j=torch.autograd.grad(sum((u*v).sum() for u,v in zip(b,vs)),y)[0]
torch.testing.assert_close(g,j,rtol=1e-4,atol=3e-5)
print('SAM 4 feature outputs + input gradients verified; max abs grad delta', (g-j).abs().max().item())
