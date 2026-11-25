"""Check SDXL-style nested UNet checkpointing and successive PCM/DMD backwards."""
import os,torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed._composable.fsdp import fully_shard,MixedPrecisionPolicy
from diffusers import UNet2DConditionModel
from diffusers.models.attention import BasicTransformerBlock
from diffusers.models.resnet import ResnetBlock2D,Downsample2D,Upsample2D
from diffusers.models.transformers.transformer_2d import Transformer2DModel
rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank);torch.set_num_threads(2)
dist.init_process_group('nccl');mesh=init_device_mesh('cuda',(dist.get_world_size(),));torch.manual_seed(10)
model=UNet2DConditionModel(sample_size=16,in_channels=4,out_channels=4,block_out_channels=(32,64,128),layers_per_block=1,down_block_types=('DownBlock2D','CrossAttnDownBlock2D','CrossAttnDownBlock2D'),up_block_types=('CrossAttnUpBlock2D','CrossAttnUpBlock2D','UpBlock2D'),transformer_layers_per_block=(1,2,2),cross_attention_dim=16,attention_head_dim=(4,8,8),norm_num_groups=8,addition_embed_type='text_time',addition_time_embed_dim=8,projection_class_embeddings_input_dim=64).cuda().float()
model.enable_gradient_checkpointing();policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16,reduce_dtype=torch.float32,cast_forward_inputs=True)
for module in reversed(list(model.modules())):
 if isinstance(module,(BasicTransformerBlock,ResnetBlock2D,Transformer2DModel,Downsample2D,Upsample2D)):fully_shard(module,mesh=mesh,mp_policy=policy)
fully_shard(model,mesh=mesh,mp_policy=policy)
opt=torch.optim.AdamW(model.parameters(),lr=1e-4,foreach=False)
x=torch.randn(1,4,16,16,device='cuda');ctx=torch.randn(1,8,16,device='cuda');t=torch.tensor([999],device='cuda');added={'text_embeds':torch.randn(1,16,device='cuda'),'time_ids':torch.tensor([[1024,1024,0,0,1024,1024]],device='cuda',dtype=torch.float32)}
def forward():
 with torch.autocast('cuda',dtype=torch.bfloat16):return model(x,t,encoder_hidden_states=ctx,added_cond_kwargs=added).sample.float()
for i in range(2):
 opt.zero_grad(set_to_none=True)
 with torch.no_grad():target=forward()
 forward().square().mean().backward()
 with torch.no_grad():target=forward()
 (forward()-target-.1).square().mean().backward()
 assert all(p.grad is None or torch.isfinite(p.grad.to_local()).all() for p in model.parameters())
 opt.step()
print('PASS FSDP2 UNet nested checkpointing + no_grad/grad transitions + two backwards + FP32 AdamW',rank,flush=True)
dist.destroy_process_group()
