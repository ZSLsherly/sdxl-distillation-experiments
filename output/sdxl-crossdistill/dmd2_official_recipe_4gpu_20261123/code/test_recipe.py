"""CPU checks: recipe, exact reference warmup, shard batch coverage, source precision contract."""
import ast,json
from pathlib import Path
import torch
from diffusers.optimization import get_scheduler
root=Path(__file__).resolve().parents[1]
cfg=json.loads((root/'config.json').read_text())
assert cfg['student_lr']==cfg['fake_lr']==5e-7
assert cfg['betas']==[.9,.999] and cfg['grad_clip']==10 and cfg['weight_decay']==.01
p=torch.nn.Parameter(torch.ones(1))
opt=torch.optim.AdamW([p],lr=5e-7,betas=(.9,.999),weight_decay=.01)
sched=get_scheduler('constant_with_warmup',optimizer=opt,num_warmup_steps=500)
for iteration in range(800):
    expected=5e-7*min(iteration*64/500,1.)
    assert abs(opt.param_groups[0]['lr']-expected)<1e-20
    # Accelerate 0.23 split_batches=False advances once per original process.
    opt.step()
    for _ in range(64):sched.step()
for world in (2,4):
    for micro in (1,2,4,8,16):
        accum=64//(world*micro)
        ids=[(m*micro+j)*world+rank for m in range(accum) for j in range(micro) for rank in range(world)]
        assert sorted(ids)==list(range(64))
source=(root/'code/train_matched.py').read_text()
ast.parse(source)
assert 'param_dtype=None,reduce_dtype=torch.float32,cast_forward_inputs=False' in source
assert 'param_dtype=torch.bfloat16' not in source
assert 'torch.backends.cuda.matmul.allow_tf32=True' in source
assert 'torch.backends.cudnn.allow_tf32=True' in source
assert 'schedule.jump(x,eps,tt,s)' not in source
assert 'schedule.noise(x0,torch.randn_like(x0),s)' in source
assert "clip(student_params,cfg['grad_clip'])" in source
assert "clip(fake_params,cfg['grad_clip'])" in source
print('RECIPE_WARMUP_BATCH_PRECISION_STATIC_PASS')
