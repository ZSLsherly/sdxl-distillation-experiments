from pathlib import Path
import json
import math
import os
import sys
import torch
from launch import flags
from train import arguments
from turning_sampling import TurningSampler
from batching import random_inputs
import boundary

root = Path(os.environ['SDXL_ROOT'])
job = root/'outputs/A_pcm10_terminal_low30_g160_20261113'
cfg = json.loads((job/'effective_config.json').read_text())
sys.argv = ['train', '--run', job.name+'/model', *flags(cfg)]
args = arguments()
assert args.pcm_weight == 10 and args.fake_updates == 2 and args.gan_weight == 0
assert getattr(args, 'fake_student_mix', 0) == 0
assert not getattr(args, 'terminal_only_dm', False)
sampler = TurningSampler(args.turning_profile, 979)
ranges = boundary.phase_ranges([999,749,499,249,-1],10,979)
rates = {}
for phase in ['student','fake']:
    for k in [2,3,4]:
        seed = args.seed + 1000*k + (500000 if phase=='fake' else 0)
        old = random_inputs(range(8), phase, 'cpu', shape=(4,48,48),
                            seed=args.seed+1000*k, time_range=ranges[k])
        before = {n:v.clone() for n,v in old.items() if isinstance(v,torch.Tensor)}
        sampler.apply(old,range(8),phase,seed,ranges[k],include_fake=True)
        for n,v in before.items():
            if n!='t' or k!=4:
                assert torch.equal(old[n],v), (phase,k,n)
        if k==4:
            t = sampler.sample(range(20000),seed,ranges[k])
            rates[phase] = sum(x<=100 for x in t)/len(t)
            assert abs(rates[phase]-.3)<.015
for g in range(160):
    assert [boundary.critic_stage(2*g+i,True,args.seed) for i in range(2)] == [boundary.middle_stage(g,args.seed),4]
_,p = sampler.distribution((20,979))
assert math.isclose(sum(p[:81]),.3,abs_tol=1e-12)
assert math.isclose(sum(p[81:]),.7,abs_tol=1e-12)
result = dict(passed=True, empirical_low30=rates, middle_random_inputs_exactly_preserved=True,
              PCM=10, EMA=0, student_DM='random middle + terminal',fake='paired middle + terminal',
              effective_batch=64, micro=16, fake_weight_max=args.fake_weight_max)
(job/'cpu_validation.json').write_text(json.dumps(result,indent=2))
print('CPU_VALIDATED', result)
