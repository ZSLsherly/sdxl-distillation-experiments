"""Validate removal of student middle loss and non-invasive three-term diagnostics."""
from pathlib import Path
import ast,importlib.util,json,os,random,sys
import torch
from diffusers import DDIMScheduler
from objectives import VPSchedule
from launch import flags
from train import arguments
import boundary as bd
from pcm_middle_probe_n13 import gradient_diagnostics, PROBE_UPDATES

root=Path(os.environ['SDXL_ROOT']);job=root/'outputs/A_pcm1_terminal_dm_g160_20261113'
base=root/'outputs/turning_BC_paired_quota_20261102/uniform_control'
cfg=json.loads((job/'effective_config.json').read_text())
sys.argv=['train','--run',job.name+'/model',*flags(cfg)]
a=arguments()
assert (a.pcm_weight,a.fake_updates,a.batch,a.micro_batch,a.max_student_updates)==(1,2,64,16,160)
assert a.turning_profile is None and a.gan_weight==0 and not getattr(a,'fake_student_mix',0)
spec=importlib.util.spec_from_file_location('baseline_boundary',base/'source/boundary.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
knots=[999,749,499,249,-1]
assert bd.phase_ranges(knots,10,979)==old.phase_ranges(knots,10,979)
sched=VPSchedule(DDIMScheduler(beta_start=.00085,beta_end=.012,beta_schedule='scaled_linear').alphas_cumprod)
w=torch.nn.Parameter(torch.tensor([.01,.02,.03,.04]));torch.manual_seed(123);z=torch.randn(2,4,4,4)
def predict(x,t,c):return w[knots.index(int(t[0]))]*x
for mid in (2,3):
    aa,ab=bd.rollout_boundary_and_final(predict,sched,None,z,knots,mid)
    ba,bb=old.rollout_boundary_and_final(predict,sched,None,z,knots,mid)
    torch.testing.assert_close(aa,ba,rtol=0,atol=0);torch.testing.assert_close(ab,bb,rtol=0,atol=0)
    for k,y in bd.local_rollout(predict,sched,None,z,knots,None):
        assert k==4;torch.testing.assert_close(y,bb)
        assert torch.autograd.grad(y.square().mean(),w)[0].nonzero().flatten().tolist()==[3]
    for k,y in bd.local_rollout(predict,sched,None,z,knots,mid):
        if k==mid:
            assert torch.autograd.grad(y.square().mean(),w)[0].nonzero().flatten().tolist()==[mid-1]
        else:assert k==4
for g in range(160):
    assert [bd.critic_stage(2*g+i,True,a.seed) for i in range(2)]==[bd.middle_stage(g,a.seed),4]

# Diagnostic math and RNG must be exact on known, non-orthogonal vectors.
p=torch.nn.Parameter(torch.tensor([.2,-.1,.4],dtype=torch.float64))
vectors={'PCM':torch.tensor([1.,2.,3.],dtype=torch.float64),
         'DM2':torch.tensor([-2.,1.,4.],dtype=torch.float64),
         'DM3':torch.tensor([2.,-3.,1.],dtype=torch.float64),
         'DM4':torch.tensor([3.,2.,-1.],dtype=torch.float64)}
def clear():p.grad=None
def norm():return p.grad.norm().item()
losses={k:(lambda k=k:(p*vectors[k]).sum()+0*torch.rand((),dtype=p.dtype)) for k in vectors}
optim=torch.optim.Adam([p],lr=.01);(p.square().sum()).backward();optim.step();clear()
before=p.detach().clone();states={k:v.clone() if torch.is_tensor(v) else v for k,v in optim.state[p].items()}
rng=torch.random.get_rng_state().clone();py_rng=random.getstate()
metrics=gradient_diagnostics(losses,clear,norm,'cpu',17)
assert p.grad is None and torch.equal(p,before) and torch.equal(torch.random.get_rng_state(),rng)
assert random.getstate()==py_rng
for k,v in states.items():
    if torch.is_tensor(v):assert torch.equal(v,optim.state[p][k])
for k,v in vectors.items():assert abs(metrics['grad_norms'][k]-v.norm().item())<1e-12
for key,cos in metrics['gradient_cosines'].items():
    left,right=key.split('_');va,vb=vectors[left],vectors[right]
    assert abs(cos-torch.dot(va,vb).item()/va.norm().item()/vb.norm().item())<1e-12
zero={k:(lambda:p.sum()*0) for k in vectors}
assert all(v is None for v in gradient_diagnostics(zero,clear,norm,'cpu',17)['gradient_cosines'].values())
try:
    gradient_diagnostics({**losses,'DM2':lambda:(_ for _ in ()).throw(RuntimeError('expected'))},clear,norm,'cpu',17)
except RuntimeError as e:assert str(e)=='expected'
else:raise AssertionError('Failure path not exercised')
assert p.grad is None and torch.equal(torch.random.get_rng_state(),rng)

train=Path('train.py').read_text();orig=(base/'source/train.py').read_text()
for name in ['fake_training_batch','boundary_loss','pcm_loss','phase_random','clip_sharded']:
    def node(text):
        return next((n for n in ast.walk(ast.parse(text)) if isinstance(n,ast.FunctionDef) and n.name==name),None)
    x,y=node(train),node(orig)
    if x is not None or y is not None:assert ast.dump(x)==ast.dump(y),name
assert 'scheduler, condition, rnd["noise"], knots, None):' in train
assert "'supervised_stages':[4]" in train and 'middle_target' not in train
assert 'scheduler,generated,knots[k],rnd' in train
assert 's = knots[boundary_dm.critic_stage(fcount,args.pair_fake_stages,args.seed)]' in train
assert 'pcm_middle_probe_n13.py' in train and 'diagnostic_seconds' in train
result=dict(passed=True,student_only_terminal_gradient=True,unchanged_fake_boundary_targets=True,
    fake_pairing_160=True,unchanged_A_ranges=True,diagnostic_math=True,
    diagnostic_rng_parameters_optimizer_unchanged=True,probe_G=PROBE_UPDATES,
    actual_SDXL_GPU_probe='first queued G1, after prior job releases GPU')
(job/'cpu_validation.json').write_text(json.dumps(result,indent=2))
print('CPU_VALIDATED',result)
