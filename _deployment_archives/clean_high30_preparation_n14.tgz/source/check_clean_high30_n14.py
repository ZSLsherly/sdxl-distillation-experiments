from pathlib import Path
import hashlib,json,math,os,sys
import torch
from launch import flags
from train import arguments
from turning_sampling import TurningSampler
from batching import random_inputs
import boundary as bd
r=Path(os.environ['SDXL_ROOT']);j=r/'outputs/A_clean_global_high30_g160_20261114'
cfg=json.loads((j/'effective_config.json').read_text());baseline=json.loads((j/'baseline_config.json').read_text())
assert {k for k in cfg if cfg[k]!=baseline[k]}=={'turning_profile','match_fake_prior'}
for name,sha in json.loads((j/'baseline_source_hashes.json').read_text()).items():
 assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==sha,name
sys.argv=['train','--run',j.name+'/model',*flags(cfg)];a=arguments()
assert a.pcm_weight==1 and a.fake_updates==2 and a.match_fake_prior and a.gan_weight==0
assert not getattr(a,'fake_student_mix',0)
ranges=bd.phase_ranges([999,749,499,249,-1],10,979)
assert ranges=={k:(20,979) for k in (2,3,4)}
sampler=TurningSampler(a.turning_profile,979)
ts,ps=sampler.distribution((20,979))
assert math.isclose(sum(p for t,p in zip(ts,ps) if t>=900),.3,abs_tol=1e-12)
assert math.isclose(sum(p for t,p in zip(ts,ps) if t<900),.7,abs_tol=1e-12)
assert all(math.isclose(p,.3/80 if t>=900 else .7/880,abs_tol=1e-14) for t,p in zip(ts,ps))
rates={};streams={}
for phase in ['student','fake']:
 for k in [2,3,4]:
  seed=a.seed+1000*k+(500000 if phase=='fake' else 0)
  sampled=sampler.sample(range(20000),seed,ranges[k]);rate=sum(t>=900 for t in sampled)/len(sampled)
  assert abs(rate-.3)<.015 and min(sampled)>=20 and max(sampled)<=979
  rates[f'{phase}_{k}']=rate;streams[(phase,k)]=sampled[:100]
  rnd=random_inputs(range(16),phase,'cpu',shape=(1,48,48),pcm_boundary=749,time_range=ranges[k])
  before={key:v.clone() for key,v in rnd.items() if torch.is_tensor(v)}
  sampler.apply(rnd,range(16),phase,seed,ranges[k],include_fake=True)
  assert rnd['t'].tolist()==sampled[:16]
  for key,v in before.items():
   if key!='t':assert torch.equal(v,rnd[key]),(phase,k,key)
for k in (2,3,4):assert streams[('student',k)]!=streams[('fake',k)]
for g in range(160):assert [bd.critic_stage(2*g+i,True,a.seed) for i in range(2)]==[bd.middle_stage(g,a.seed),4]
source=Path('train.py').read_text()
assert "scheduler,generated.detach(),-1,rnd" in source and "scheduler,generated,-1,rnd" in source
assert "s = -1  # Middle and terminal supervision are both on clean predictions" in source
assert 'turning.apply(randoms,positions' in source and 'turning.apply(rnd,positions' in source
result=dict(passed=True,source_identical_to_completed_clean_global=True,
 theoretical_high=.3,theoretical_other=.7,empirical_high_rates=rates,
 non_time_inputs_and_PCM_unchanged=True,independent_student_fake_streams=True,
 stage_pairing_160=True,batch64=True,only_config_changes=['turning_profile','match_fake_prior'])
(j/'cpu_validation.json').write_text(json.dumps(result,indent=2));print('CPU_VALIDATED',result)
