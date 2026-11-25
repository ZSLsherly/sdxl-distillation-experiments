import os,json,torch
import torch.distributed as dist
rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank)
dist.init_process_group('nccl')
x=torch.ones(1024*1024,device='cuda');dist.all_reduce(x);assert (x==4).all()
a=torch.randn(1024,1024,device='cuda',dtype=torch.bfloat16);assert torch.isfinite(a@a).all()
dist.barrier(device_ids=[rank]);print(json.dumps({'rank':rank,'passed':True,'GPU':torch.cuda.get_device_name(rank)}),flush=True)
dist.destroy_process_group()
