import os
import torch
import torch.distributed as dist
torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
dist.init_process_group('nccl')
x=torch.tensor(float(dist.get_rank()),device='cuda')
dist.all_reduce(x,op=dist.ReduceOp.MAX)
torch.cuda.synchronize()
assert x.item()==3
if dist.get_rank()==0:print('FOUR_GPU_EXIT_SYNC_OK',flush=True)
dist.barrier()
dist.destroy_process_group()
