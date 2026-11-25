"""Small CPU-only DTensor optimizer equivalence and timing check."""
import os,time,json,statistics
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor,Shard
torch.set_num_threads(4)
dist.init_process_group('gloo')
mesh=init_device_mesh('cpu',(dist.get_world_size(),))
torch.manual_seed(67)
base=[torch.randn(32,256) for _ in range(128)] + [torch.randn(512,2048) for _ in range(8)]
def params():
 return [torch.nn.Parameter(DTensor.from_local(x.clone(),mesh,[Shard(0)],run_check=False)) for x in base]
a,b=params(),params()
oa=torch.optim.AdamW(a,lr=1e-6,betas=(.9,.999),weight_decay=.01,foreach=False)
ob=torch.optim.AdamW(b,lr=1e-6,betas=(.9,.999),weight_decay=.01,foreach=True)
times={False:[],True:[]}
for step in range(6):
 grads=[torch.randn_like(x) for x in base]
 for ps,opt,mode in [(a,oa,False),(b,ob,True)] if step%2==0 else [(b,ob,True),(a,oa,False)]:
  for p,g in zip(ps,grads):p.grad=DTensor.from_local(g.clone(),mesh,[Shard(0)],run_check=False)
  t=time.perf_counter();opt.step();elapsed=time.perf_counter()-t
  if step:times[mode].append(elapsed)
 for x,y in zip(a,b):
  torch.testing.assert_close(x.to_local(),y.to_local(),rtol=1e-6,atol=1e-7)
  for k in ['exp_avg','exp_avg_sq']:torch.testing.assert_close(oa.state[x][k].to_local(),ob.state[y][k].to_local(),rtol=1e-6,atol=1e-7)
print(json.dumps({'rank':dist.get_rank(),'local_elements':sum(x.numel() for x in base),'max_parameter_delta':max((x.to_local()-y.to_local()).abs().max().item() for x,y in zip(a,b)),'seconds':{str(k):v for k,v in times.items()},'median_seconds':{str(k):statistics.median(v) for k,v in times.items()}}),flush=True)
dist.barrier();dist.destroy_process_group()
