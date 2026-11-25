import os,time,json,gc,statistics
from pathlib import Path
from datetime import timedelta
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed._composable.fsdp import fully_shard,MixedPrecisionPolicy
from diffusers import UNet2DConditionModel
from diffusers.models.resnet import ResnetBlock2D,Downsample2D,Upsample2D
from diffusers.models.transformers.transformer_2d import Transformer2DModel
from diffusers.models.attention import BasicTransformerBlock
rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank);torch.set_num_threads(2)
dist.init_process_group('nccl',timeout=timedelta(seconds=90));mesh=init_device_mesh('cuda',(2,))
root=Path('/data1/shenglin.zeng/teacher_turning_20261026');rows=[]
def emit(row):
    rows.append(row)
    print(json.dumps(dict(rank=rank,p2p_disable=os.environ['NCCL_P2P_DISABLE'],**row)),flush=True)
for mb in (1,16,64):
    x=torch.ones(mb*1024*1024//2,device='cuda',dtype=torch.bfloat16);out=torch.empty(x.numel()*2,device='cuda',dtype=x.dtype)
    for i in range(3):dist.all_gather_into_tensor(out,x)
    torch.cuda.synchronize();start=time.perf_counter()
    for i in range(10):dist.all_gather_into_tensor(out,x)
    torch.cuda.synchronize();emit(dict(kind='allgather',MiB=mb,seconds=(time.perf_counter()-start)/10))
    del x,out
for mode in ('replicated','coarse','fine'):
    torch.manual_seed(123)
    model=UNet2DConditionModel.from_pretrained(root/'models/sdxl-base-1.0/unet',variant='fp16',torch_dtype=torch.bfloat16,local_files_only=True).cuda().eval().requires_grad_(False)
    if mode!='replicated':
        policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16,reduce_dtype=torch.float32,cast_forward_inputs=False)
        if mode=='fine':
            for child in reversed(list(model.modules())):
                if isinstance(child,(ResnetBlock2D,Downsample2D,Upsample2D,Transformer2DModel,BasicTransformerBlock)):
                    fully_shard(child,mesh=mesh,mp_policy=policy)
        fully_shard(model,mesh=mesh,mp_policy=policy)
    for batch in (1,2,4):
        torch.manual_seed(222)
        x=torch.randn(batch,4,128,128,device='cuda');c=torch.randn(batch,77,2048,device='cuda');p=torch.randn(batch,1280,device='cuda');ids=torch.tensor([[1024,1024,0,0,1024,1024]],device='cuda').repeat(batch,1)
        def forward():
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                a=model(x,777,encoder_hidden_states=c,added_cond_kwargs={'text_embeds':p,'time_ids':ids}).sample
                b=model(x,777,encoder_hidden_states=torch.zeros_like(c),added_cond_kwargs={'text_embeds':torch.zeros_like(p),'time_ids':ids}).sample
                return (b.float()+8*(a.float()-b.float()))
        forward();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();times=[]
        for i in range(3):
            dist.barrier();torch.cuda.synchronize();start=time.perf_counter();y=forward();torch.cuda.synchronize();times.append(time.perf_counter()-start)
        ref=root/'tmp'/f'side_teacher_reference_rank{rank}_b{batch}.pt'
        if mode=='replicated':torch.save(y.cpu(),ref)
        delta=(y.cpu()-torch.load(ref,weights_only=True)).abs().max().item()
        emit(dict(kind='teacher_cfg8',mode=mode,batch=batch,seconds=statistics.median(times),seconds_per_sample=statistics.median(times)/batch,peak_GiB=torch.cuda.max_memory_allocated()/1024**3,max_abs_delta=delta))
        del x,c,p,ids,y
    del model;gc.collect();torch.cuda.empty_cache();dist.barrier()
dist.destroy_process_group()
