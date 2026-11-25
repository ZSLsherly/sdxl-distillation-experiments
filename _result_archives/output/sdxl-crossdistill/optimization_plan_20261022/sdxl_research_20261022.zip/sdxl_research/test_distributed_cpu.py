"""Run with torchrun --standalone --nproc_per_node=2; CPU/Gloo only."""
import json, time, statistics
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard
from runtime import average_gradients

torch.set_num_threads(2)
dist.init_process_group('gloo')
rank=dist.get_rank();world=dist.get_world_size()
torch.manual_seed(7)
model=torch.nn.Sequential(torch.nn.Linear(16,32),torch.nn.Linear(32,4))
for p in model.parameters():p.grad=torch.randn_like(p)+rank
expected=[p.grad.clone() for p in model.parameters()]
for g in expected:dist.all_reduce(g);g.div_(world)
calls=average_gradients(model.parameters(),16)
for p,g in zip(model.parameters(),expected):torch.testing.assert_close(p.grad,g,rtol=0,atol=0)
mesh=init_device_mesh('cpu',(world,))
base=[torch.randn(32,256) for _ in range(32)]
def parameters():return [torch.nn.Parameter(DTensor.from_local(x.clone(),mesh,[Shard(0)],run_check=False)) for x in base]
a,b=parameters(),parameters()
oa=torch.optim.AdamW(a,lr=1e-6,foreach=False);ob=torch.optim.AdamW(b,lr=1e-6,foreach=True)
elapsed={'scalar':[],'foreach':[]}
for step in range(8):
    grads=[torch.randn_like(x) for x in base]
    pairs=[(a,oa,'scalar'),(b,ob,'foreach')]
    if step%2:pairs.reverse()
    for params,opt,name in pairs:
        for p,g in zip(params,grads):p.grad=DTensor.from_local(g.clone(),mesh,[Shard(0)],run_check=False)
        tick=time.perf_counter();opt.step();elapsed[name].append(time.perf_counter()-tick)
    for p,q in zip(a,b):
        torch.testing.assert_close(p.to_local(),q.to_local(),rtol=1e-6,atol=1e-7)
        for key in ('exp_avg','exp_avg_sq'):
            torch.testing.assert_close(oa.state[p][key].to_local(),ob.state[q][key].to_local(),rtol=1e-6,atol=1e-7)
print(json.dumps({'rank':rank,'bucket_collectives':calls,'result':'passed',
 'adam_max_abs_difference':max((p.to_local()-q.to_local()).abs().max().item() for p,q in zip(a,b)),
 'toy_cpu_adam_median_seconds':{k:statistics.median(v[2:]) for k,v in elapsed.items()},
 'scope':'CPU DTensor arithmetic and Gloo only; not FSDP CUDA or end-to-end speed proof'}),flush=True)
dist.destroy_process_group()
