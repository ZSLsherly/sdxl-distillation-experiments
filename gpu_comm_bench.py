import os,time,json,datetime
import torch
import torch.distributed as dist
rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank)
dist.init_process_group('nccl',timeout=datetime.timedelta(seconds=90))
world=dist.get_world_size()
results=[]
for mib in [1,16,64,256]:
 x=torch.ones(mib*1024*1024//4,device='cuda')
 for _ in range(5): dist.all_reduce(x)
 torch.cuda.synchronize();dist.barrier();start=time.perf_counter()
 for _ in range(20):
  x.fill_(rank+1);dist.all_reduce(x)
 torch.cuda.synchronize();elapsed=(time.perf_counter()-start)/20
 assert x[0].item()==world*(world+1)/2
 value=torch.tensor(elapsed,device='cuda');dist.all_reduce(value,op=dist.ReduceOp.MAX)
 if rank==0:
  r={'MiB':mib,'latency_ms':value.item()*1000,'algorithm_GB_s':mib*1024**2/value.item()/1e9,'bus_GB_s':mib*1024**2/value.item()/1e9*2*(world-1)/world};results.append(r);print(json.dumps(r),flush=True)
if rank==0:
 print('P2P_ACCESS',[[torch.cuda.can_device_access_peer(i,j) if i!=j else True for j in range(world)] for i in range(world)],flush=True)
 open('/data5/shenglin.zeng/benchmarks/nccl.json','w').write(json.dumps(results,indent=2))
dist.destroy_process_group()
