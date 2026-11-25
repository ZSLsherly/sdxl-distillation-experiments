import sys,json,ast,math,random
from pathlib import Path
root=Path.cwd();out=root/'output/sdxl-crossdistill/project_audit_20261112/evidence'
sys.path[:0]=[str(root/'tmp/boundary_cpu_runtime'),str(root/'sdxl_research')]
import torch
import boundary as bd
from objectives import VPSchedule
from fake_tracking import verify_tracking_probes
from turning_sampling import TurningSampler
from unittest.mock import patch
betas=torch.linspace(math.sqrt(.00085),math.sqrt(.012),1000,dtype=torch.float32).square()
sched=VPSchedule((1-betas).cumprod(0));weights=(10*sched.sigma.square()).clamp_max(1)
result={'fake_loss_terminal_weights':{str(t):weights[t].item() for t in [20,40,60,80,96,100,900]},'mean_low20_100_weight':weights[20:101].mean().item()}
# Compare true error coefficient with the claimed epsilon MSE, including gradients.
y=torch.zeros(4,1,1,1);t=torch.tensor([20,60,100,940]);noise=torch.ones_like(y);pred=torch.zeros_like(y,requires_grad=True)
l=bd.fake_loss(sched,y,-1,t,noise,pred,10);g=torch.autograd.grad(l,pred)[0]
expected=(weights[t].reshape(-1,1,1,1)*(pred-noise).square()).mean()
torch.testing.assert_close(l,expected);result['terminal_weight_algebra']=True
sampler=TurningSampler(root/'sdxl_research/turning_profiles/teacher_interval_quota_C_979.json',979)
p=sampler.profile.copy();p['terminal_low_only']=True;p['probabilities']=[.3/81 if t<=100 else .7/879 for t in p['timesteps']]
p.pop('terminal_interval_masses',None);file=out/'low30_audit_profile.json';file.write_text(json.dumps(p))
source=(root/'sdxl_research/turning_sampling.py').read_text();source=source.replace('        times=list(range(lo,hi+1))',"        times=list(range(lo,hi+1))\n        if self.profile.get('terminal_low_only') and (lo,hi)==(20,979):\n            return times,self.profile['probabilities']")
ns={};exec(compile(source,'patched_turning_sampling','exec'),ns);new=ns['TurningSampler'](file,979)
assert new.distribution((509,979))==sampler.distribution((509,979));assert new.distribution((259,979))==sampler.distribution((259,979))
ts,prob=new.distribution((20,979));assert math.isclose(sum(p for t,p in zip(ts,prob) if t<=100),.3,abs_tol=1e-12)
result['low30_changes_terminal_only']=True
# Reproduce the old verifier rejecting the valid logging contract.
ids=list(range(8));points=[dict(time=t,before=.1,after=.09,direction_cosine=.99,relative_direction_change=.1) for t in (60,499,940)]
rows=[dict(phase='heldout_fake_tracking',g=g,original_ids=ids,student_mix=.01,results=points) for g in sorted([1,*range(8,161,8),121])]
verify_tracking_probes(rows,ids,160,1,.01)
assert [r['g'] for r in rows if r['phase']=='fake_student_mix']!=list(range(1,161))
result['old_queue_rejects_valid_probe_contract']=True
bad=rows[:-1]
try:verify_tracking_probes(bad,ids,160,1,.01)
except ValueError:result['new_verifier_rejects_missing_probe']=True
else:raise AssertionError('missing probe accepted')
for p in [root/'tmp/run_terminal_low30.py',root/'tmp/prepare_terminal_low30.py',root/'tmp/test_terminal_low30.py']:compile(p.read_text(encoding='utf-8-sig'),str(p),'exec')
# Standard entry point already prohibits resume with tracking; do not report an unreachable resume issue.
from train import arguments
from launch import flags
cfg=json.loads((root/'sdxl_research/configs/C_mix1_pcm_terminal_dm.json').read_text());cfg['resume']='dummy';cfg['matched_wall_seconds']=None
with patch('sys.argv',['train','--run','audit',*flags(cfg)]):
 try:arguments()
 except SystemExit as e:assert e.code!=0;result['tracking_resume_rejected_at_entry']=True
 else:raise AssertionError('unexpected accepted resume')
(out/'audit_reproductions.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
