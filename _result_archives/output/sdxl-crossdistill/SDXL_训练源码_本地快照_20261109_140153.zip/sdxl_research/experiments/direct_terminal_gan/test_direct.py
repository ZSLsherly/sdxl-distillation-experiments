"""CPU gradient regression for the H20 training helpers, no models or downloads."""
import sys
from pathlib import Path
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from boundary import local_rollout, rollout_boundary_and_final
from gan_losses import generator_loss, discriminator_loss
from terminal_gan import terminal_input, backward_terminal

knots=[999,749,499,249,-1]
class Schedule:
    def jump(self,x,eps,t,next_t):
        return x-eps

def student_grad(middle, refine_route, accum, weight):
    params=torch.nn.ParameterList([torch.nn.Parameter(torch.tensor(i/10.)) for i in range(1,6)])
    head=torch.nn.Linear(1,1,bias=False).requires_grad_(False)
    head.weight.fill_(2)
    calls=[]
    probes=[]
    def predict(x,t,c):
        k=knots.index(int(t[0]))
        calls.append(k)
        return x*params[k]
    def refine(x):
        assert not x.requires_grad
        calls.append(4)
        return x*params[4]
    for batch in torch.arange(1,9,dtype=torch.float).reshape(-1,1).chunk(accum):
        for stage,x in local_rollout(predict,Schedule(),{},batch,knots,middle):
            if stage==4:
                gan=backward_terminal(x.square().mean(),x,refine if refine_route else None,
                    lambda z:generator_loss(head(3*z)),weight,accum,
                    lambda grad:probes.append(grad.detach().clone()))
                assert not gan.requires_grad
    assert calls==(list(range(5)) if refine_route else list(range(4)))*accum
    assert all(p.grad is None for p in params[:3]) and head.weight.grad is None
    for grad in probes:
        torch.testing.assert_close(grad,torch.full_like(grad,-weight*6/8))
    assert len(probes)==accum and (params[4].grad is not None)==refine_route
    return torch.tensor([p.grad.item() if p.grad is not None else 0 for p in params[3:]])

for mid in (2,3):
    for route in (True,False):
        grad=student_grad(mid,route,4,.005)
        torch.testing.assert_close(grad,student_grad(mid,route,1,.005))
        delta=grad-student_grad(mid,route,4,0)
        x249=.9*.8*.7*4.5
        expected=[0,-.005*6*x249*.6] if route else [.005*6*x249,0]
        torch.testing.assert_close(delta,torch.tensor(expected),atol=2e-6,rtol=1e-5)

student=torch.nn.Parameter(torch.tensor(.1))
head=torch.nn.Linear(1,1,bias=False)
head.weight.data.fill_(.1)
for refine in (None,lambda z:z*.2):
    with torch.no_grad():
        _,x0=rollout_boundary_and_final(lambda x,t,c:x*student,Schedule(),{},torch.ones(4,1),knots,4)
        negative=terminal_input(x0,refine)
    torch.testing.assert_close(negative,x0 if refine is None else x0*.2)
    head.zero_grad()
    discriminator_loss(head(torch.ones_like(negative)),head(negative)).backward()
    assert student.grad is None and head.weight.grad.abs().item()>0
print('PASS: D negatives and gradient isolation; G direct/refined, middle 2/3, 4/5 calls, GAN-only hook, accumulated gradients')
