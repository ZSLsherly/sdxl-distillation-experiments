import os,torch,torch.distributed as dist
from datetime import timedelta
from flash_attn import flash_attn_func
rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank)
dist.init_process_group('nccl',timeout=timedelta(seconds=90))
x=torch.tensor([rank+1.],device='cuda');dist.all_reduce(x)
assert x.item()==10,x
q=torch.randn(1,32,8,64,device='cuda',dtype=torch.bfloat16)
y=flash_attn_func(q,q,q)
assert y.shape==q.shape and torch.isfinite(y).all()
torch.cuda.synchronize();dist.barrier()
print('GPU_PREFLIGHT_PASS',rank,torch.cuda.get_device_name(),torch.__version__,flush=True)
dist.destroy_process_group()
