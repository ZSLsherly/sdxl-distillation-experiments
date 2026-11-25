"""Small CPU DTensor optimizer diagnostic; no SDXL load, no training-quality claim."""
import json,socket,time,traceback
from datetime import timedelta
from pathlib import Path
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor,Shard
torch.set_num_threads(4);torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(.1)
s=socket.socket();s.bind(('127.0.0.1',0));port=s.getsockname()[1];s.close()
dist.init_process_group('nccl',rank=0,world_size=1,init_method=f'tcp://127.0.0.1:{port}',timeout=timedelta(seconds=30))
mesh=init_device_mesh('cuda',(1,));torch.manual_seed(3)
initial=[torch.randn(32+(i%5)*8,64) for i in range(300)]
grads=[torch.randn_like(v) for v in initial]
result={};states={}
for foreach in [False,True]:
 try:
  ps=[torch.nn.Parameter(DTensor.from_local(v.cuda(),mesh,[Shard(0)],run_check=False).cpu()) for v in initial]
  assert all(p.to_local().device.type=='cpu' for p in ps)
  opt=torch.optim.AdamW(ps,lr=5e-6,betas=(0.,.999),weight_decay=0,foreach=foreach)
  times=[];norm_times=[]
  for step in range(3):
   for p,g in zip(ps,grads):p.grad=DTensor.from_local(g.cuda(),mesh,[Shard(0)],run_check=False).cpu()
   assert all(p.grad.to_local().device.type=='cpu' for p in ps)
   tick=time.monotonic()
   norms=torch.stack(torch._foreach_norm([p.grad.to_local() for p in ps],2))
   total=norms.double().square().sum().cuda();dist.all_reduce(total);norm=total.sqrt().item()
   torch._foreach_mul_([p.grad.to_local() for p in ps],min(1.,1/(norm+1e-12)))
   norm_times.append(time.monotonic()-tick)
   tick=time.monotonic();opt.step();times.append(time.monotonic()-tick)
   assert all(v['exp_avg'].to_local().device.type=='cpu' for v in opt.state.values())
   opt.zero_grad(set_to_none=True)
   print('STEP',foreach,step,times[-1],flush=True)
  states[foreach]=[p.to_local().detach().clone() for p in ps]
  result[str(foreach)]={'optimizer_seconds':times,'norm_clip_seconds':norm_times,'parameters':sum(v.numel() for v in initial),'verified_parameter_gradient_moment_device':'cpu'}
  del opt,ps
 except Exception as e:result[str(foreach)]={'error':repr(e)};traceback.print_exc()
if False in states and True in states:
 result['max_abs_parameter_difference']=max((x-y).abs().max().item() for x,y in zip(states[False],states[True]))
result['scope']='CPU local tensors wrapped as DTensor on one CUDA mesh, 300 small tensors; not full SDXL throughput'
Path('/data1/shenglin.zeng/teacher_turning_20261026/outputs/turning_boundary_ablation_20261027/optimizer_probe_cpu.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2),flush=True);dist.destroy_process_group()
