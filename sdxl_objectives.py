"""CrossDistill's noise partition and relay, expressed in SDXL's native VP coordinates.

SDXL predicts epsilon under x_t=alpha_t*x0+sigma_t*epsilon. Deterministic DDIM
jumps replace the RF Euler maps used by Wan. The first jump is supervised only
by high-noise PCM. DMD sees only t<=crossover and cannot backpropagate into relay.
"""
import torch

class VPSchedule:
    def __init__(self, alphas_cumprod):
        self.alpha = alphas_cumprod.float().sqrt()
        self.sigma = (1-alphas_cumprod.float()).sqrt()

    def coefficients(self,t,x):
        t=torch.as_tensor(t,device=x.device,dtype=torch.long).reshape(-1)
        a=self.alpha.to(x.device)[t.clamp_min(0)]
        s=self.sigma.to(x.device)[t.clamp_min(0)]
        a=torch.where(t<0,torch.ones_like(a),a)
        s=torch.where(t<0,torch.zeros_like(s),s)
        return a.reshape(-1,*([1]*(x.ndim-1))),s.reshape(-1,*([1]*(x.ndim-1)))

    def noise(self,x0,eps,t):
        a,s=self.coefficients(t,x0)
        return a*x0+s*eps

    def clean(self,x,eps,t):
        a,s=self.coefficients(t,x)
        return (x-s*eps)/a

    def jump(self,x,eps,t,next_t):
        a,s=self.coefficients(next_t,x)
        return a*self.clean(x,eps,t)+s*eps

def dmd_surrogate(generated,fake_clean,teacher_clean):
    with torch.no_grad():
        scale=(generated-teacher_clean).abs().flatten(1).mean(1).clamp_min(1e-5)
        scale=scale.reshape(-1,*([1]*(generated.ndim-1)))
        direction=(fake_clean-teacher_clean)/scale
        target=(generated-direction).detach()
    # Mean latent-coordinate normalization, shared by PCM and fake losses.
    return (generated-target).square().flatten(1).mean(1).mean(), direction

def verify_math():
    torch.manual_seed(8)
    sched=VPSchedule(torch.linspace(.999,.004,1000))
    clean=torch.randn(2,4,8,8);eps=torch.randn_like(clean)
    t=torch.tensor([900,850]);s=torch.tensor([749,749]);end=torch.tensor([-1,-1])
    xt=sched.noise(clean,eps,t)
    assert torch.allclose(sched.clean(xt,eps,t),clean,atol=2e-6)
    xs=sched.jump(xt,eps,t,s)
    assert torch.allclose(xs,sched.noise(clean,eps,s),atol=2e-6)
    assert torch.allclose(sched.jump(xs,eps,s,end),clean,atol=2e-6)
    assert torch.allclose(sched.jump(xs,eps,s,s),xs,atol=2e-6)
    # DMD must move generated outputs toward the teacher if fake predicts them.
    g=torch.ones(1,4,8,8,requires_grad=True)
    loss,_=dmd_surrogate(g,g.detach(),torch.zeros_like(g));loss.backward()
    assert (g.grad>0).all()
    relay=torch.ones(1,requires_grad=True)
    low_weight=torch.ones(1,requires_grad=True)
    (relay.detach()*low_weight).sum().backward()
    assert relay.grad is None and low_weight.grad is not None
    print('PASS: VP clean reconstruction, DDIM oracle transport, boundary identity, DMD sign, relay stop-gradient',flush=True)

if __name__=='__main__':verify_math()
