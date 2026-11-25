"""Four-rank save/load/next-update equivalence for the real checkpoint helper."""
import os
import sys
from pathlib import Path
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed._composable.fsdp import fully_shard, CPUOffloadPolicy, MixedPrecisionPolicy
from training_state import save_training, load_training, cpu_state, optimizer_state

rank = int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(rank)
dist.init_process_group('nccl')
mesh = init_device_mesh('cuda', (dist.get_world_size(),))
policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)

def setup():
    models, opts = {}, {}
    for name in ('student', 'fake', 'discriminator'):
        m = torch.nn.Sequential(torch.nn.Linear(16, 32), torch.nn.SiLU(), torch.nn.Linear(32, 16)).cuda()
        if name != 'discriminator':
            extra = {'offload_policy': CPUOffloadPolicy()} if name == 'fake' else {}
            fully_shard(m, mesh=mesh, mp_policy=policy, **extra)
        models[name] = m
        opts[name] = torch.optim.AdamW(m.parameters(), lr=1e-6, foreach=False)
    return models, opts

def step(models, opts):
    for name, model in models.items():
        x = torch.randn(4, 16, device='cuda')
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss = model(x).float().square().mean()
        loss.backward()
        if name == 'discriminator':
            for p in model.parameters():
                dist.all_reduce(p.grad)
                p.grad.div_(dist.get_world_size())
        opts[name].step()
        opts[name].zero_grad(set_to_none=True)

torch.manual_seed(1)
models, opts = setup()
step(models, opts)
path = Path(sys.argv[1])
save_training(path, models, opts, {'gcount': 1, 'fcount': 5, 'dcount': 5}, {})
step(models, opts)
expected = {k: cpu_state(m) for k,m in models.items()}
expected_opts = {k: optimizer_state(o) for k,o in opts.items()}
models, opts = setup()
progress = load_training(path, models, opts, {})
assert progress == {'gcount': 1, 'fcount': 5, 'dcount': 5}
step(models, opts)
for name, model in models.items():
    for key, value in cpu_state(model).items():
        torch.testing.assert_close(value, expected[name][key], rtol=0, atol=0)
    got = optimizer_state(opts[name])
    for key, values in got['state'].items():
        for field, value in values.items():
            torch.testing.assert_close(value, expected_opts[name]['state'][key][field], rtol=0, atol=0)
print('PASS_SAVE_LOAD_NEXT_UPDATE_EXACT', rank, flush=True)
dist.barrier()
dist.destroy_process_group()
