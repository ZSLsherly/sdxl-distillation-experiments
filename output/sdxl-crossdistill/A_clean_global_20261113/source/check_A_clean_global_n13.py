"""CPU checks for actual deployed sampler, clean target and local gradients."""
from pathlib import Path
import importlib.util
import json
import os
import sys
import torch
from diffusers import DDIMScheduler
from objectives import VPSchedule
from batching import random_inputs
from launch import flags
from train import arguments
import boundary as bd

root=Path(os.environ['SDXL_ROOT']);job=root/'outputs/A_clean_global_g160_20261113'
cfg=json.loads((job/'effective_config.json').read_text())
sys.argv=['train','--run',job.name+'/model',*flags(cfg)]
args=arguments()
assert (args.pcm_weight,args.fake_updates,args.gan_weight)==(1,2,0)
assert args.turning_profile is None and args.pair_fake_stages and args.fake_weight_max==10
knots=[999,749,499,249,-1]
ranges=bd.phase_ranges(knots,10,979)
assert ranges=={k:(20,979) for k in (2,3,4)}
old_file=root/'outputs/A_mid_clean_x0/model/source/boundary.py'
spec=importlib.util.spec_from_file_location('old_boundary',old_file)
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
assert old.phase_ranges(knots,10,979)=={2:(509,979),3:(259,979),4:(20,979)}
rates={}
for phase in ['student','fake']:
    rnd=random_inputs(range(2048),phase,'cpu',shape=(1,48,48),time_range=ranges[2])
    t=rnd['t']; assert t.min()>=20 and t.max()<=979 and (t<259).any()
    histogram=torch.histc(t.float(),bins=4,min=20,max=980)/len(t)
    assert (histogram-.25).abs().max()<.04
    rates[phase]=dict(low_fraction=(t<=100).float().mean().item(),histogram=histogram.tolist())
    # Packing and non-time inputs are unchanged, including PCM noise and times.
    for k in (2,3,4):
        before=random_inputs(range(8),phase,'cpu',shape=(1,48,48),time_range=old.phase_ranges(knots,10,979)[k])
        after=random_inputs(range(8),phase,'cpu',shape=(1,48,48),time_range=ranges[k])
        for key in before:
            if isinstance(before[key],torch.Tensor) and (key!='t' or k==4):
                torch.testing.assert_close(before[key],after[key],rtol=0,atol=0)
sched=VPSchedule(DDIMScheduler(beta_start=.00085,beta_end=.012,beta_schedule='scaled_linear').alphas_cumprod)
torch.manual_seed(17)
y=torch.randn(4,4,4,4,requires_grad=True);eps=torch.randn_like(y);t=torch.tensor([20,60,499,979])
xt=bd.renoise(sched,y,-1,t,eps)
torch.testing.assert_close(xt,sched.noise(y,eps,t))
ef=torch.randn_like(y);et=torch.randn_like(y)
loss,direction=bd.dm_surrogate(sched,y,-1,t,xt.detach(),ef,et)
with torch.no_grad():
    pf=sched.clean(xt,ef,t);pt=sched.clean(xt,et,t)
    ref=(pf-pt)/(y-pt).abs().flatten(1).mean(1).clamp_min(1e-5).reshape(-1,1,1,1)
torch.testing.assert_close(direction,ref,rtol=2e-4,atol=2e-5)
torch.testing.assert_close(torch.autograd.grad(loss,y)[0],ref/y.numel(),rtol=2e-4,atol=2e-6)
_,sigma=sched.coefficients(t,y)
floss=bd.fake_loss(sched,y.detach(),-1,t,eps,ef,10.)
torch.testing.assert_close(floss,((ef-eps).square()*(10*sigma.square()).clamp_max(1)).mean())
w=torch.nn.Parameter(torch.tensor([.01,.02,.03,.04]));z=torch.randn(2,4,4,4)
def predict(x,t,c):return w[knots.index(int(t[0]))]*x
for mid in (2,3):
    a,b=bd.rollout_boundary_and_final(predict,sched,None,z,knots,mid)
    aa,bb=old.rollout_boundary_and_final(predict,sched,None,z,knots,mid)
    torch.testing.assert_close(a,aa);torch.testing.assert_close(b,bb)
    for k,out in bd.local_rollout(predict,sched,None,z,knots,mid):
        grad=torch.autograd.grad(out.square().mean(),w)[0]
        assert grad.nonzero().flatten().tolist()==[k-1]
for g in range(160):
    assert [bd.critic_stage(2*g+i,True,args.seed) for i in range(2)]==[bd.middle_stage(g,args.seed),4]
train=Path('train.py').read_text()
assert "scheduler,generated.detach(),-1,rnd" in train and "scheduler,generated,-1,rnd" in train
assert "s = -1  # Middle and terminal supervision are both on clean predictions" in train
assert train.count('time_range=ranges')>=3
result=dict(passed=True,ranges=ranges,sampling=rates,clean_renoise=True,
    standard_DM_direction_and_gradient=True,unchanged_fake_weighting=True,
    unchanged_rollout_and_non_time_random_inputs=True,local_gradient_and_fake_pairing=True)
(job/'cpu_validation.json').write_text(json.dumps(result,indent=2))
print('CPU_VALIDATED',result)
