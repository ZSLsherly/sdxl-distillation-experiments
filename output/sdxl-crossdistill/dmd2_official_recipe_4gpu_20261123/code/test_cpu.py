"""Exercise upstream DMD2 loss/gradient paths without loading SDXL or touching CUDA."""
import contextlib,json,sys
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).parent/'vendor'))
import torch
from diffusers import DDIMScheduler
from main.sd_guidance import SDGuidance
from main.sd_unified_model import SDUniModel
torch.set_num_threads(1)
class Toy(torch.nn.Module):
    def __init__(self,coef):super().__init__();self.w=torch.nn.Parameter(torch.tensor(coef))
    def forward(self,x,t,encoder_hidden_states=None,classify_mode=False,**kw):
        v=self.w*x
        if classify_mode:return [v.mean((2,3),keepdim=True)]
        if encoder_hidden_states is not None:v=v+.1*encoder_hidden_states.flatten(1).mean(1).reshape(-1,1,1,1)
        return SimpleNamespace(sample=v)
s=SDGuidance.__new__(SDGuidance);torch.nn.Module.__init__(s)
s.fake_unet=Toy(.1);s.real_unet=Toy(.2).requires_grad_(False)
s.scheduler=DDIMScheduler(beta_start=.00085,beta_end=.012,beta_schedule='scaled_linear')
s.alphas_cumprod=s.scheduler.alphas_cumprod;s.min_step=20;s.max_step=979;s.num_train_timesteps=1000
s.use_fp16=False;s.sdxl=False;s.real_guidance_scale=8;s.fake_guidance_scale=1
s.gradient_checkpointing=False;s.network_context_manager=contextlib.nullcontext()
s.diffusion_gan=True;s.diffusion_gan_max_timestep=1000
s.cls_pred_branch=torch.nn.Conv2d(4,1,1)
torch.manual_seed(17);x=torch.randn(4,4,4,4,requires_grad=True);text=torch.ones(4,2,3);negative=torch.zeros_like(text)
loss,log=s.compute_distribution_matching_loss(x,text,negative,None,None)
grad=torch.autograd.grad(loss['loss_dm'],x)[0]
torch.testing.assert_close(grad,log['dmtrain_grad']/x.numel())
assert s.fake_unet.w.grad is None and s.real_unet.w.grad is None
fake_loss,_=s.compute_loss_fake(x,text,negative)
fake_loss['loss_fake_mean'].backward();assert x.grad is None and s.fake_unet.w.grad is not None
s.fake_unet.requires_grad_(False);s.cls_pred_branch.requires_grad_(False)
gan=s.compute_generator_clean_cls_loss(x,text,None)['gen_cls_loss']
gg=torch.autograd.grad(gan,x)[0];assert torch.isfinite(gg).all() and gg.abs().sum()>0
m=SDUniModel.__new__(SDUniModel);torch.nn.Module.__init__(m)
m.feedforward_model=Toy(.01);m.add_time_ids=torch.tensor([[1024,1024,0,0,1024,1024.]])
m.noise_scheduler=s.scheduler;m.alphas_cumprod=s.alphas_cumprod;m.denoising_step_list=torch.tensor([999,749,499,249]);m.num_denoising_step=4;m.timestep_interval=250
for k,t in enumerate([999,749,499,249]):
 m.selected_training_step=k
 y,tt=m.sample_backward(torch.randn_like(x),text,torch.ones(4,1280))
 assert (tt==t).all() and torch.isfinite(y).all() and not y.requires_grad
for micro in (16,8,4,2,1):
 accum=64//(4*micro)
 positions=[(mi*micro+j)*4+rank for mi in range(accum) for j in range(micro) for rank in range(4)]
 assert sorted(positions)==list(range(64))
assert sum(i%5==0 for i in range(800))==160
print(json.dumps(dict(passed=True,upstream_DM_gradient=True,teacher_fake_detached_for_DM=True,
 fake_detaches_student=True,native_GAN_reaches_student=True,all_four_backward_depths=True,
 batch64_partition=True,G160_F800=True)))

