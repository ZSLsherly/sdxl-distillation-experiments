import socket,time
from datetime import timedelta
import torch
import torch.distributed as dist
torch.set_num_threads(2);torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(.1)
sock=socket.socket();sock.bind(('127.0.0.1',0));port=sock.getsockname()[1];sock.close()
print('IMPORT_AND_CUDA_OK',flush=True);start=time.monotonic()
dist.init_process_group('nccl',rank=0,world_size=1,init_method=f'tcp://127.0.0.1:{port}',timeout=timedelta(seconds=30))
print('PROCESS_GROUP_OK',flush=True)
x=torch.ones(1,device='cuda');dist.all_reduce(x);torch.cuda.synchronize()
print('ALL_REDUCE_OK',x.item(),time.monotonic()-start,flush=True)
dist.barrier();dist.destroy_process_group();print('DONE',flush=True)
