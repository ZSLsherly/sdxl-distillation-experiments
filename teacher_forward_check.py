import json,time,torch
from imaginaire.utils import distributed
from sparkdiffusion.datasets.build_synthetic_dataset import _load_dit
distributed.init()
net=_load_dit('1.3B','/data5/shenglin.zeng/models/Wan2.1-T2V-1.3B')
x=torch.randn(1,16,21,60,104,device='cuda',dtype=torch.bfloat16)
context=torch.zeros(1,512,4096,device='cuda',dtype=torch.bfloat16)
t=torch.full((1,1),940.,device='cuda',dtype=torch.bfloat16)
torch.cuda.reset_peak_memory_stats()
times=[]
with torch.inference_mode():
 for i in range(3):
  torch.cuda.synchronize();start=time.perf_counter()
  y=net(x_B_C_T_H_W=x,timesteps_B_T=t,crossattn_emb=context)
  torch.cuda.synchronize();times.append(time.perf_counter()-start)
  assert y.shape==x.shape and torch.isfinite(y).all()
result={'kind':'teacher_forward_with_zero_text_for_hardware_preflight_only','shape':list(x.shape),'seconds':times,'peak_allocated_GiB':torch.cuda.max_memory_allocated()/1024**3}
print(json.dumps(result),flush=True)
open('/data5/shenglin.zeng/benchmarks/teacher_forward.json','w').write(json.dumps(result,indent=2))
torch.distributed.destroy_process_group()
