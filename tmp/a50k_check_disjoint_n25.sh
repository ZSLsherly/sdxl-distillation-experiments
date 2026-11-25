set -eu
cd /data5/shenglin.zeng/a50k_20261124/code
CUDA_VISIBLE_DEVICES='' /data5/shenglin.zeng/envs/crossdistill/bin/python - <<'PY'
import ast,inspect,json,sys
from pathlib import Path
from types import SimpleNamespace
import torch,train,boundary
from launch import flags
from batching import random_inputs
cfg=json.loads(Path('base.json').read_text())
cfg.update(data_dir='datasets/parallel_50k_seed20261018',max_student_updates=784,
           checkpoint_mode='full',snapshot_seconds=10**8,hours=168,boundary_noise_max=None)
sys.argv=['train.py','--run','A50K_disjoint',*flags(cfg)]
args=train.arguments()
assert args.boundary_noise_max is None and args.pair_fake_stages and not args.terminal_only_dm
spans=boundary.phase_ranges([*args.schedule,-1],args.boundary_margin,args.boundary_noise_max)
assert spans=={2:(509,739),3:(259,489),4:(20,239)}
for phase in ['fake','student']:
 for k,(lo,hi) in spans.items():
  rnd=random_inputs(list(range(512)),phase,'cpu',shape=(1,48,48),time_range=(lo,hi),stream=k if phase=='student' else 0)
  assert all(lo<=t<=hi for t in rnd['t'].tolist())
  print(phase,k,'bounds',lo,hi,'observed',rnd['t'].min().item(),rnd['t'].max().item())
src=ast.parse(inspect.getsource(train.main))
fn=next(n for n in ast.walk(src) if isinstance(n,ast.FunctionDef) and n.name=='predict')
# Check the actual kwargs construction for a fake forward without loading network weights.
guard=next(n for n in ast.walk(fn) if isinstance(n,ast.If) and 'class_labels' in ast.unparse(n) and isinstance(n.test,ast.BoolOp) and 'model is fake' in ast.unparse(n.test))
fake=object()
for stage in [2,3,4]:
 scope=dict(model=fake,fake=fake,args=args,score_stage=stage,extra={})
 exec(compile(ast.Module(body=[guard],type_ignores=[]),'predict_kwargs','exec'),scope)
 assert 'class_labels' not in scope['extra']
print('PASS_DISJOINT_AND_NO_FAKE_CLASS_LABELS')
PY
