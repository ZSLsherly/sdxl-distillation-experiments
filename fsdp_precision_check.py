import os,torch,json
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed._composable.fsdp import fully_shard,MixedPrecisionPolicy
torch.cuda.set_device(int(os.environ['LOCAL_RANK']));dist.init_process_group('nccl')
torch.manual_seed(42)
net=torch.nn.Sequential(torch.nn.Linear(64,128),torch.nn.GELU(),torch.nn.Linear(128,64)).cuda()
mesh=init_device_mesh('cuda',(4,))
policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16,reduce_dtype=torch.float32)
for m in [net[0],net[2],net]:fully_shard(m,mesh=mesh,mp_policy=policy)
optimizer=torch.optim.AdamW(net.parameters(),lr=2e-6)
p=next(net.parameters());before=p.to_local().detach().clone()
x=torch.randn(4,64,device='cuda',dtype=torch.bfloat16)
y=net(x);assert y.dtype==torch.bfloat16
y.float().square().mean().backward()
assert p.dtype==torch.float32 and p.grad.dtype==torch.float32
optimizer.step()
change=(p.to_local().detach()-before).abs().max().item()
assert change>0
print(json.dumps({'rank':dist.get_rank(),'output_dtype':str(y.dtype),'master_dtype':str(p.dtype),'gradient_dtype':str(p.grad.dtype),'parameter_delta':change}),flush=True)
dist.destroy_process_group()
