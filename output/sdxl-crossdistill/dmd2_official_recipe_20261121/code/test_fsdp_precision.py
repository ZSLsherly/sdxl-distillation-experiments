"""Small two-GPU check: FSDP2 preserves FP32 no-autocast and BF16 autocast outputs/gradients."""
import os,copy,json
from pathlib import Path
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed._composable.fsdp import fully_shard,MixedPrecisionPolicy
dist.init_process_group('nccl')
rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank);torch.set_num_threads(2)
torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
torch.manual_seed(123)
ref=torch.nn.Sequential(torch.nn.Linear(32,64),torch.nn.SiLU(),torch.nn.Linear(64,32)).cuda()
net=copy.deepcopy(ref);mesh=init_device_mesh('cuda',(dist.get_world_size(),))
policy=MixedPrecisionPolicy(param_dtype=None,reduce_dtype=torch.float32,cast_forward_inputs=False)
fully_shard(net[0],mesh=mesh,mp_policy=policy);fully_shard(net[2],mesh=mesh,mp_policy=policy)
fully_shard(net,mesh=mesh,mp_policy=policy)
x=torch.randn(4,32,device='cuda');proof=[]
for amp in (False,True):
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16,enabled=amp):
        expected=ref(x);actual=net(x)
    assert actual.dtype==(torch.bfloat16 if amp else torch.float32)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    for model in (ref,net):
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=amp):loss=model(x).float().square().mean()
        loss.backward()
    for a,b in zip(net.parameters(),ref.parameters()):
        assert a.dtype==a.grad.dtype==torch.float32
        torch.testing.assert_close(a.grad.full_tensor(),b.grad,rtol=1e-5,atol=1e-7)
    proof.append({'autocast':amp,'output':str(actual.dtype),'forward_exact':True,'gradient_match':True})
    ref.zero_grad(set_to_none=True);net.zero_grad(set_to_none=True)
if rank==0:
    dest=Path(__file__).resolve().parents[1]/'fsdp_precision_passed.json'
    dest.write_text(json.dumps(proof,indent=2));print('FSDP_PRECISION_OUTPUT_GRADIENT_PASS',flush=True)
dist.barrier();dist.destroy_process_group()
