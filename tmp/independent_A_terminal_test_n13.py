"""Independent small-network integration test using deployed objective modules."""
from pathlib import Path
import copy,importlib.util,json,os,tempfile
import torch
from diffusers import DDIMScheduler
import boundary as bd
from objectives import VPSchedule
from pcm_middle_probe_n13 import gradient_diagnostics

torch.set_num_threads(1)
sched=VPSchedule(DDIMScheduler(beta_start=.00085,beta_end=.012,beta_schedule='scaled_linear').alphas_cumprod)
knots=[999,749,499,249,-1];ranges=bd.phase_ranges(knots,10,979)
class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__();self.w=torch.nn.Parameter(torch.randn(3)*.01)
        self.stage=torch.nn.Parameter(torch.zeros(3))
    def forward(self,x,t,stage=4):
        tt=t.reshape(-1,1,1,1).to(x)/1000
        return self.w[0]*x+self.w[1]*tt+self.w[2]*torch.tanh(x)+self.stage[stage-2]
def teacher(x,t):return .03*x+.02*t.reshape(-1,1,1,1)/1000
def pcm(student,real,noise,th):
    xh=sched.noise(real,noise,th)
    with torch.no_grad():
        xk=xh
        for k in range(2):
            tk=th-k*10;xk=sched.jump(xk,teacher(xk,tk),tk,tk-10)
        target=sched.jump(xk,student(xk,th-20),th-20,749)
    return (sched.jump(xh,student(xh,th),th,749)-target).square().mean()
def dm(fake,y,k,t,noise):
    with torch.no_grad():
        xt=bd.renoise(sched,y.detach(),knots[k],t,noise)
        ef=fake(xt,t,k);et=teacher(xt,t)
    return bd.dm_surrogate(sched,y,knots[k],t,xt,ef,et)[0]
def state(obj):
    if torch.is_tensor(obj):return obj.detach().clone()
    if isinstance(obj,dict):return {k:state(v) for k,v in obj.items()}
    if isinstance(obj,list):return [state(v) for v in obj]
    return copy.deepcopy(obj)
def equal(a,b):
    if torch.is_tensor(a):assert torch.equal(a,b),(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:equal(a[k],b[k])
    elif isinstance(a,list):
        assert len(a)==len(b)
        for x,y in zip(a,b):equal(x,y)
    else:assert a==b,(a,b)

def run(probes,manual=False):
    torch.manual_seed(71);student=Tiny();fake=Tiny()
    og=torch.optim.AdamW(student.parameters(),lr=1e-6,betas=(0,.999),weight_decay=0)
    of=torch.optim.AdamW(fake.parameters(),lr=5e-6,betas=(0,.999),weight_decay=0)
    pair=[];clips=[]
    for g in range(3):
        mid=bd.middle_stage(g,20261017)
        for i in range(2):
            k=bd.critic_stage(2*g+i,True,20261017);pair.append(k)
            noise=torch.randn(8,1,3,3)
            y,_=bd.rollout_boundary_and_final(lambda x,t,c:student(x,t),sched,None,noise,knots,k)
            tt=torch.randint(ranges[k][0],ranges[k][1]+1,(8,));eps=torch.randn_like(y)
            of.zero_grad(set_to_none=True)
            for start in range(0,8,2):
                ys,ts,es=y[start:start+2],tt[start:start+2],eps[start:start+2]
                xt=bd.renoise(sched,ys,knots[k],ts,es)
                (bd.fake_loss(sched,ys,knots[k],ts,es,fake(xt,ts,k),10)/4).backward()
            torch.nn.utils.clip_grad_norm_(fake.parameters(),1);of.step();of.zero_grad(set_to_none=True)
        if probes:
            before=state((student.state_dict(),fake.state_dict(),og.state_dict(),of.state_dict()))
            with torch.random.fork_rng():
                torch.manual_seed(100);z=torch.randn(8,1,3,3);real=torch.randn_like(z);pn=torch.randn_like(z)
                th=torch.randint(770,1000,(8,))
                times={k:torch.randint(ranges[k][0],ranges[k][1]+1,(8,)) for k in (2,3,4)}
                noises={k:torch.randn_like(z) for k in (2,3,4)}
                def dl(k):
                    stream=bd.local_rollout(lambda x,t,c:student(x,t),sched,None,z,knots,k if k<4 else None)
                    try:
                        for st,y in stream:
                            if st==k:return dm(fake,y,k,times[k],noises[k])
                    finally:stream.close()
                losses={'PCM':lambda:pcm(student,real,pn,th),**{f'DM{k}':lambda k=k:dl(k) for k in (2,3,4)}}
                gradient_diagnostics(losses,lambda:og.zero_grad(set_to_none=True),
                    lambda:torch.linalg.vector_norm(torch.cat([p.grad.flatten() for p in student.parameters() if p.grad is not None])).item(),'cpu',17)
            after=state((student.state_dict(),fake.state_dict(),og.state_dict(),of.state_dict()))
            # Tuple wrapper handled explicitly, avoiding Python tensor truth values.
            for a,b in zip(before,after):equal(a,b)
            assert all(p.grad is None for p in student.parameters())
        # Fresh global RNG input makes any diagnostic RNG leak observable.
        z=torch.randn(8,1,3,3);real=torch.randn_like(z);pn=torch.randn_like(z)
        th=torch.randint(770,1000,(8,));tt=torch.randint(20,980,(8,));eps=torch.randn_like(z)
        og.zero_grad(set_to_none=True)
        for start in range(0,8,2):
            zz=z[start:start+2]
            if manual:
                x=zz
                for k in range(1,5):
                    t=torch.full((len(x),),knots[k-1])
                    with torch.set_grad_enabled(k==4):x=sched.jump(x,student(x,t),t,knots[k])
                    if k<4:x=x.detach()
                (dm(fake,x,4,tt[start:start+2],eps[start:start+2])/4).backward()
            else:
                for k,y in bd.local_rollout(lambda x,t,c:student(x,t),sched,None,zz,knots,None):
                    assert k==4
                    (dm(fake,y,k,tt[start:start+2],eps[start:start+2])/4).backward()
            (pcm(student,real[start:start+2],pn[start:start+2],th[start:start+2])/4).backward()
        clips.append(float(torch.nn.utils.clip_grad_norm_(student.parameters(),1)))
        og.step();og.zero_grad(set_to_none=True)
    return dict(student=state(student.state_dict()),fake=state(fake.state_dict()),
                optim_student=state(og.state_dict()),optim_fake=state(of.state_dict()),
                rng=torch.random.get_rng_state(),pair=pair,clips=clips)

plain=run(False);probed=run(True);reference=run(False,manual=True)
equal(plain,probed);equal(plain,reference)
result=dict(passed=True,three_student_six_fake_updates=True,
    diagnostic_on_off_bitwise_identical_student_fake_Adam_RNG=True,
    manual_terminal_only_reference_bitwise_identical=True,pair=plain['pair'],
    scope='CPU shared-parameter small network; real deployed VP/boundary/diagnostic modules, not SDXL CUDA/FSDP')
root=Path(os.environ['SDXL_ROOT']);job=root/'outputs/A_pcm1_terminal_dm_g160_20261113'
(job/'independent_integration_test.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
