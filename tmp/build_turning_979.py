"""Matched raw-angle / noise-normalized turning priors from the SAME 512 trajectories."""
import hashlib,json,html
from pathlib import Path
import numpy as np

base=Path.cwd()
source=base/'output/sdxl-crossdistill/teacher_turning_resample_20261026/results512/raw'
out=base/'output/sdxl-crossdistill/turning_ablation_20261027';out.mkdir(exist_ok=True)
profiles=base/'sdxl_research/turning_profiles';profiles.mkdir(exist_ok=True)
files=sorted(source.glob('base_dt10_bf16_*.json'));assert len(files)==512
h=hashlib.sha256();data=[]
for f in files:
 raw=f.read_bytes();h.update(f.name.encode());h.update(raw);data.append(json.loads(raw))
rows=data[0]['rows'];u=np.array([r['u_mid'] for r in rows]);t=np.array([r['t_mid'] for r in rows])
assert len({(d['prompt_index'],d['seed']) for d in data})==512
for d in data:assert np.array_equal([r['u_mid'] for r in d['rows']],u)
angles=np.array([[r['fields']['rf_full']['angle_rad'] for r in d['rows']] for d in data])
du=np.array([r['delta_u'] for r in rows]);order=np.argsort(u)
ugrid=np.linspace(u.min(),u.max(),256);bandwidth=.015
kernel=np.exp(-.5*((ugrid[:,None]-ugrid[None,:])/bandwidth)**2);kernel/=kernel.sum(1,keepdims=True)
times=np.arange(20,980);u_at_t=np.interp(times,t[::-1],u[::-1]);beta=.3
curves={'normalized_u':(angles/du).mean(0),'raw_angle':angles.mean(0)}
summary={};ps={}
for name,curve in curves.items():
 uniform_curve=np.interp(ugrid,u[order],curve[order]);smooth=kernel@uniform_curve
 rates=np.interp(u_at_t,ugrid,smooth);offset=.1*np.median(rates)
 weight=(rates+offset)**2;prior=weight/weight.sum();q=(1-beta)/len(times)+beta*prior
 p={'name':name,'metric':'theta / abs(delta_u)' if name=='normalized_u' else 'theta (no noise/path division)',
    'beta':beta,'support':[20,979],'timesteps':times.tolist(),'probabilities':q.tolist(),
    'prior_probabilities':prior.tolist(),'profile_values':rates.tolist(),'offset':float(offset),
    'offset_rule':'0.1 * median of each smoothed profile on [20,979]', 'exponent':2,
    'source_sha256':h.hexdigest(),'trajectories':512,'prompts':128,'seeds':[201,202,203,204],
    'smoothing':'Interpolate onto 256 uniform physical-u points, Gaussian bandwidth .015, then interpolate onto native integer t; same for both arms',
    'importance_correction':False,'state_field':'rf_full, C=identity, CFG8, BF16, native stride10',
    'probability_measure':'mass on integer native timesteps; not a density uniform in u'}
 (profiles/f'{name}_512_979.json').write_text(json.dumps(p,indent=2),encoding='utf-8');ps[name]=p
 summary[name]={'mean_t':float(times@q),'mass_low_20_199':float(q[times<200].sum()),
     'mass_middle_200_799':float(q[(times>=200)&(times<800)].sum()),'mass_high_800_919':float(q[times>=800].sum()),
     'max_to_uniform_ratio':float(q.max()*900),'probability_sum':float(q.sum())}

import sys
sys.path.insert(0,str(base/'sdxl_research'))
from turning_sampling import TurningSampler
for name in curves:
 old=json.loads((profiles/f'{name}_512.json').read_text())
 new=json.loads((profiles/f'{name}_512_979.json').read_text())
 assert old['source_sha256']==new['source_sha256']
 np.testing.assert_allclose(old['profile_values'],new['profile_values'][:900],rtol=0,atol=1e-12)
 sampler=TurningSampler(profiles/f'{name}_512_979.json',979)
 for stage,r in {2:(509,979),3:(259,979),4:(20,979)}.items():
  tt,pp=sampler.distribution(r);assert abs(sum(pp)-1)<1e-12
  draws=sampler.sample(range(2000),20261017,time_range=r)
  assert min(draws)>=r[0] and max(draws)<=r[1] and max(draws)>919
  print(name,stage,'mean',sum(t*p for t,p in zip(tt,pp)),'P(t>919)',sum(p for t,p in zip(tt,pp) if t>919))
