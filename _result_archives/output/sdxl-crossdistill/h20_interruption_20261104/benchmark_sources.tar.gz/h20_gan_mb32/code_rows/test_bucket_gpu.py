"""Tiny 2-rank NCCL check: bucketed gradients and next optimizer update."""
import copy,json,os
import torch
import torch.distributed as dist
from runtime import average_gradients

def main():
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group('nccl')
    torch.manual_seed(11)
    a=torch.nn.Sequential(torch.nn.Linear(16,64),torch.nn.SiLU(),torch.nn.Linear(64,4)).cuda()
    b=copy.deepcopy(a)
    for p,q in zip(a.parameters(),b.parameters()):
        p.grad=torch.randn_like(p)+dist.get_rank();q.grad=p.grad.clone()
    scalar=average_gradients(a.parameters(),0)
    bucket=average_gradients(b.parameters(),16)
    for p,q in zip(a.parameters(),b.parameters()):torch.testing.assert_close(p.grad,q.grad,rtol=0,atol=0)
    oa=torch.optim.AdamW(a.parameters(),lr=1e-6,foreach=False)
    ob=torch.optim.AdamW(b.parameters(),lr=1e-6,foreach=False)
    oa.step();ob.step()
    for p,q in zip(a.parameters(),b.parameters()):torch.testing.assert_close(p,q,rtol=0,atol=0)
    print(json.dumps({'rank':dist.get_rank(),'result':'exact_gradient_and_next_update',
                      'scalar_collectives':scalar,'bucket_collectives':bucket}),flush=True)
    dist.destroy_process_group()


if __name__=='__main__':
    main()
