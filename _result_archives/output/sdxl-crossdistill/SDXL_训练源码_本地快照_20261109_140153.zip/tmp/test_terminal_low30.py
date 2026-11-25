from pathlib import Path
import os,sys,json,math,torch
from launch import flags
from train import arguments
import boundary as bd
from turning_sampling import TurningSampler
from batching import random_inputs
job=Path(os.environ['SDXL_ROOT'])/'outputs/C_mix1_pcm_dm4_fake4_low30_g160'
cfg=json.loads((job/'effective_config.json').read_text());sys.argv=['train','--run','validation',*flags(cfg)];a=arguments()
assert a.terminal_only_dm and a.matched_wall_seconds is None and a.fake_updates==2
assert [bd.critic_stage(i,True,a.seed,2) for i in range(320)]==[k for g in range(160) for k in (bd.middle_stage(g,a.seed),4)]
sampler=TurningSampler(a.turning_profile,979);ts,p=sampler.distribution((20,979))
assert math.isclose(sum(v for t,v in zip(ts,p) if t<=100),.3,abs_tol=1e-12)
assert math.isclose(sum(v for t,v in zip(ts,p) if t>100),.7,abs_tol=1e-12)
base_cfg=json.loads((Path(os.environ['SDXL_ROOT'])/'outputs/bc_winner_late_mix_20261102/effective_config.json').read_text())
original=TurningSampler(Path(os.environ['SDXL_ROOT'])/base_cfg['turning_profile'],979)
for r in [(509,979),(259,979)]:assert sampler.distribution(r)==original.distribution(r)
assert a.fake_weight_max==10
rates={} 
for phase,seed in [('student',a.seed+4000),('fake',a.seed+504000)]:
 rnd={'t':torch.zeros(20000,dtype=torch.long)}
 sampler.apply(rnd,range(20000),phase,seed,(20,979),include_fake=True)
 rates[phase]=(rnd['t']<=100).float().mean().item()
 assert abs(rates[phase]-.3)<.015
 assert rnd['t'].min()>=20 and rnd['t'].max()<=979
from test_boundary import BoundaryTests
import unittest
suite=unittest.TestSuite([BoundaryTests('test_clean_boundary_reduces_to_dmd2_and_epsilon_regression'),BoundaryTests('test_prefix_is_detached_and_fake_captures_actual_boundary')])
assert unittest.TextTestRunner().run(suite).wasSuccessful()
# All three routing sites use the same patched function: rollout input, fake loss, and logging.
s=Path('train.py').read_text();assert s.count('boundary_dm.critic_stage(')>=3
assert 'student_middle = () if args.terminal_only_dm' in s
(job/'cpu_validation.json').write_text(json.dumps({'passed':True,'prior_empirical':rates,'fake_stages':['selected middle',4],'student_dm_stages':[4],'pcm_weight':a.pcm_weight},indent=2))
print('CPU_VALIDATED',rates)
