import sys,json,math,hashlib,collections
from pathlib import Path
import numpy as np
import torch
from diffusers import DDIMScheduler
sys.path.insert(0,'/root/sdxl_research/code/sdxl_research')
import boundary as bd
from objectives import VPSchedule
from turning_sampling import TurningSampler
from batching import BatchPlan
root=Path('/root/sdxl_research');out=root/'outputs/logic_audit_20261101';out.mkdir(exist_ok=True)
schedule=VPSchedule(DDIMScheduler.from_pretrained(root/'models/sdxl-base-1.0/scheduler',local_files_only=True).alphas_cumprod)
records=[]
# Independent Gaussian oracle: noising an arbitrary boundary Gaussian.
nodes,weights=np.polynomial.hermite.hermgauss(12);nodes=torch.tensor(nodes*np.sqrt(2),dtype=torch.float64);weights=torch.tensor(weights/np.sqrt(np.pi),dtype=torch.float64)
z,e=torch.meshgrid(nodes,nodes,indexing='ij');ww=(weights[:,None]*weights[None,:]).reshape(-1,1)
for s,t in [(499,509),(499,900),(499,979),(249,259),(249,900),(249,979),(-1,20),(-1,100),(-1,900),(-1,979)]:
 aa=1. if s==-1 else float(schedule.alpha[s]);ss=0. if s==-1 else float(schedule.sigma[s]);at=float(schedule.alpha[t]);st=float(schedule.sigma[t]);a=at/aa;b2=st*st-a*a*ss*ss
 assert b2>0
 mu,mt,v,vt=.6,-.2,.5,1.3
 y=(mu+math.sqrt(v)*z).reshape(-1,1).requires_grad_();noise=e.reshape(-1,1)
 xt=bd.renoise(schedule,y.detach(),s,t,noise)
 torch.testing.assert_close(xt,a*y.detach()+math.sqrt(b2)*noise,rtol=3e-5,atol=1e-6)
 qv=a*a*v+b2;pv=a*a*vt+b2
 sq=-(xt-a*mu)/qv;sp=-(xt-a*mt)/pv
 loss,direction=bd.dm_surrogate(schedule,y,s,t,xt,-st*sq,-st*sp)
 grad=torch.autograd.grad(loss,y)[0];torch.testing.assert_close(grad,direction/y.numel(),rtol=2e-5,atol=1e-6)
 posterior=(xt+b2*sp)/a;scale=(y.detach()-posterior).abs().clamp_min(1e-5)
 raw=direction*scale*a*a/b2
 def kl(m):return .5*(math.log(pv/qv)+(qv+(a*m-a*mt)**2)/pv-1)
 h=1e-5;finite=(kl(mu+h)-kl(mu-h))/(2*h);observed=(ww*raw).sum().item()
 assert abs(observed-finite)<2e-5,(s,t,observed,finite)
 # Conditional noise regression optimum yields marginal score, including sigma_t/b.
 x=.8;noise_mean=math.sqrt(b2)*(x-a*mu)/qv
 pred=st/math.sqrt(b2)*noise_mean
 assert abs(-pred/st+(x-a*mu)/qv)<1e-10
 records.append({'boundary':s,'time':t,'variance':b2,'KL_finite_difference':finite,'score_gradient_mean':observed})
run=root/'outputs/turning_BC_interval_quota_20261101/B'
rows=[json.loads(s) for s in (run/'rank0.jsonl').read_text().splitlines()]
config=json.loads((run/'config.json').read_text());sampler=TurningSampler(Path(config['turning_profile']),979);plan=BatchPlan(64,1,16)
matched=0;counts=collections.defaultdict(list)
for r in rows:
 if r['phase']=='boundary_turning_sample':
  expected=sampler.sample(plan.positions(r['g'],r['micro'],0),config['seed']+1000*r['stage'],time_range=bd.phase_ranges([999,749,499,249,-1],10,979)[r['stage']])
  assert expected==r['timesteps'];matched+=len(expected);counts[r['stage']]+=expected
all_expected={}
for arm in ['B','C']:
 smp=TurningSampler(root/f'code/sdxl_research/turning_profiles/teacher_interval_quota_{arm}_979.json',979);v=collections.defaultdict(list)
 for g in range(160):
  for k in [bd.middle_stage(g,20261017),4]:v[k]+=smp.sample(range(g*64,(g+1)*64),20261017+1000*k,time_range=bd.phase_ranges([999,749,499,249,-1],10,979)[k])
 all_expected[arm]={k:{'n':len(ts),'high_fraction':sum(900<=t<=979 for t in ts)/len(ts),'low_fraction':sum(20<=t<=100 for t in ts)/len(ts)} for k,ts in v.items()}
result={'analytic_oracle_cases':records,'live_timesteps_exactly_reproduced':matched,'live_observed':{k:{'n':len(v),'high_fraction':sum(900<=t<=979 for t in v)/len(v)} for k,v in counts.items()},'deterministic_G160_sampling_plan':all_expected,'note':'CPU analytic tests and deterministic RNG replay; not a quality or convergence guarantee'}
(out/'results.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
