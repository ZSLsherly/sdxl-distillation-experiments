"""Analytic score checks and real autograd tests; CPU, no pretrained weights."""
import copy
import contextlib
import io
import json
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
import torch
from torch import nn
from diffusers import DDIMScheduler
from objectives import VPSchedule, dmd_surrogate
import boundary as bd
from batching import random_inputs, rollout_chunks
from image_views import decode_view
from sam_pixel import SAMDiscriminator
from gan_losses import discriminator_loss, generator_loss


class BoundaryTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(31)
        self.sched = VPSchedule(DDIMScheduler(beta_start=.00085,beta_end=.012,
            beta_schedule='scaled_linear',clip_sample=False).alphas_cumprod)
        self.knots = [999,749,499,249,-1]

    def test_ranges_and_packing_independent_times(self):
        self.assertEqual(bd.phase_ranges(self.knots),{2:(509,739),3:(259,489),4:(20,239)})
        for stage in (2,3,4):
            r=bd.phase_ranges(self.knots)[stage]
            all_r=random_inputs(range(6),'fake','cpu',shape=(4,48,48),time_range=r)
            split=[random_inputs(range(i,i+2),'fake','cpu',shape=(4,48,48),time_range=r) for i in (0,2,4)]
            for key in ('noise','eps','t'):
                torch.testing.assert_close(all_r[key],torch.cat([x[key] for x in split]),rtol=0,atol=0)
            self.assertTrue(((all_r['t']>=r[0])&(all_r['t']<=r[1])).all())
        self.assertEqual([bd.critic_stage(i) for i in range(6)],[2,3,4,2,3,4])
        self.assertEqual([bd.middle_stage(i,3) for i in range(8)],
                         [bd.middle_stage(i,3) for i in range(8)])

    def test_clean_boundary_reduces_to_dmd2_and_epsilon_regression(self):
        y=torch.randn(4,4,3,3,requires_grad=True);noise=torch.randn_like(y)
        t=torch.tensor([20,85,150,239]);ef=torch.randn_like(y);et=torch.randn_like(y)
        x=bd.renoise(self.sched,y,-1,t,noise)
        torch.testing.assert_close(x,self.sched.noise(y,noise,t))
        loss,direction=bd.dm_surrogate(self.sched,y,-1,t,x,ef,et)
        old,old_dir=dmd_surrogate(y,self.sched.clean(x,ef,t),self.sched.clean(x,et,t))
        torch.testing.assert_close(direction,old_dir,rtol=2e-5,atol=2e-6)
        torch.testing.assert_close(loss,.5*old,rtol=2e-5,atol=2e-6)
        torch.testing.assert_close(bd.fake_loss(self.sched,y,-1,t,noise,ef,float('inf')),
                                  (ef-noise).square().mean())

    def test_forward_kernel_preserves_teacher_VP_marginal(self):
        # Independent marginal check: start from true p_s, advance by our kernel,
        # and compare mean/variance to directly noising clean data at u.
        for s,u in ((499,509),(499,739),(249,259),(249,489),(-1,20),(-1,239)):
            template=torch.zeros(1,1)
            a,b,_=bd.coefficients(self.sched,s,u,template)
            aa,ss=self.sched.coefficients(s,template)
            au,su=self.sched.coefficients(u,template)
            mu,variance=.7,1.4
            torch.testing.assert_close(a*aa*mu,au*mu)
            torch.testing.assert_close(a.square()*(aa.square()*variance+ss.square())+b.square(),
                                       au.square()*variance+su.square())

    def test_conditional_target_integrates_to_analytic_gaussian_score(self):
        # Integrate E[noise|x_u] under its analytic conditional Gaussian.
        # 12-point quadrature is exact for this linear target, unlike a flaky MC check.
        nodes,weights=np.polynomial.hermite.hermgauss(12)
        z=torch.tensor(nodes*np.sqrt(2),dtype=torch.float64)
        w=torch.tensor(weights/np.sqrt(np.pi),dtype=torch.float64)
        for s,u in ((499,509),(499,700),(249,400),(-1,200)):
            a,b,sigma=bd.coefficients(self.sched,s,u,torch.zeros(1,1,dtype=torch.float64))
            a,b,sigma=(v.double().item() for v in (a,b,sigma))
            mu,var,x=.7,.4,1.2
            variance=a*a*var+b*b
            noise_mean=b/variance*(x-a*mu)
            noise_std=(1-b*b/variance)**.5
            noise=noise_mean+noise_std*z
            prediction=sigma/b*noise_mean
            # fake_loss optimum: expected derivative is zero at correct score.
            pred=torch.full((12,1),prediction,dtype=torch.float64,requires_grad=True)
            _,bb,ss=bd.coefficients(self.sched,s,u,pred)
            residual=bb*pred-ss*noise[:,None]
            objective=(w[:,None]*residual.square()/bb.square()).sum()
            gradient=torch.autograd.grad(objective,pred)[0].sum()
            self.assertLess(abs(gradient.item()),2e-5)
            expected_score=-(x-a*mu)/variance
            self.assertAlmostEqual(-(w*(sigma/b*noise)).sum().item()/sigma,expected_score,places=10)
            if s>=0:
                # Negative control: naive epsilon target has a biased optimum.
                self.assertGreater(abs(-noise_mean/sigma-expected_score),1e-3)

    def test_score_direction_matches_KL_finite_difference_including_a(self):
        # E_y,noise a*(score_q-score_p) equals d KL(q_u||p_u)/d mean(y_s).
        nodes,weights=np.polynomial.hermite.hermgauss(10)
        z=torch.tensor(nodes*np.sqrt(2),dtype=torch.float64)
        weights=torch.tensor(weights/np.sqrt(np.pi),dtype=torch.float64)
        yy,ee=torch.meshgrid(z,z,indexing='ij');ww=(weights[:,None]*weights[None,:]).reshape(-1,1)
        mu,mu_t,var,var_t=.6,-.2,.5,1.3
        y=(mu+var**.5*yy).reshape(-1,1).requires_grad_();noise=ee.reshape(-1,1)
        s,u=499,700
        a,b,sigma=bd.coefficients(self.sched,s,u,y)
        a,b,sigma=a.double(),b.double(),sigma.double()
        x=(a*y+b*noise).detach();qv=a*a*var+b*b;pv=a*a*var_t+b*b
        sq=-(x-a*mu)/qv;sp=-(x-a*mu_t)/pv
        loss,direction=bd.dm_surrogate(self.sched,y,s,u,x,-sigma*sq,-sigma*sp)
        posterior=(x+b*b*sp)/a
        scale=(y.detach()-posterior).abs().clamp_min(1e-5)
        raw=direction*scale*a*a/(b*b)
        def kl(m):
            return .5*(torch.log(pv/qv)+(qv+(a*m-a*mu_t)**2)/pv-1).item()
        h=1e-5;finite=(kl(mu+h)-kl(mu-h))/(2*h)
        self.assertAlmostEqual((ww*raw).sum().item(),finite,places=5)
        grad=torch.autograd.grad(loss,y)[0]
        torch.testing.assert_close(grad,direction/y.numel())

    def test_generator_fake_teacher_gradient_isolation(self):
        y=torch.randn(2,4,3,3,requires_grad=True);noise=torch.randn_like(y);t=torch.tensor([600,700])
        f=nn.Conv2d(4,4,1);teacher=nn.Conv2d(4,4,1)
        x=bd.renoise(self.sched,y.detach(),499,t,noise)
        ef=f(x);et=teacher(x)
        loss,_=bd.dm_surrogate(self.sched,y,499,t,x,ef,et)
        loss.backward()
        self.assertGreater(y.grad.norm().item(),0)
        self.assertTrue(all(p.grad is None for m in (f,teacher) for p in m.parameters()))
        y.grad=None
        bd.fake_loss(self.sched,y,499,t,noise,f(x)).backward()
        self.assertIsNone(y.grad)
        self.assertTrue(all(p.grad is not None for p in f.parameters()))

    def test_local_backward_matches_detached_summed_loss_and_chunking(self):
        class Student(nn.Module):
            def __init__(self):super().__init__();self.w=nn.Parameter(torch.tensor(.03));self.b=nn.Parameter(torch.tensor(.01))
            def forward(self,x,t,c):return self.w*x+self.b
        original=Student();noise=torch.randn(4,4,3,3)
        for middle in (2,3):
            expected=copy.deepcopy(original);actual=copy.deepcopy(original);packed=copy.deepcopy(original)
            losses=[x.square().mean() for k,x in bd.local_rollout(expected,self.sched,{},noise,self.knots,middle)]
            sum(losses).backward()
            for k,x in bd.local_rollout(actual,self.sched,{},noise,self.knots,middle):
                x.square().mean().backward()
            for chunk in noise.split(2):
                for k,x in bd.local_rollout(packed,self.sched,{},chunk,self.knots,middle):
                    (x.square().mean()/2).backward()
            for p,q,r in zip(expected.parameters(),actual.parameters(),packed.parameters()):
                torch.testing.assert_close(p.grad,q.grad)
                torch.testing.assert_close(p.grad,r.grad)

    def test_prefix_is_detached_and_fake_captures_actual_boundary(self):
        weights=[nn.Parameter(torch.tensor(.01)) for _ in range(4)]
        calls=[]
        def model(x,t,c):
            k=self.knots.index(int(t[0]));calls.append((k,torch.is_grad_enabled()))
            return weights[k]*x
        noise=torch.randn(2,4,3,3)
        for middle in (2,3):
            for p in weights:p.grad=None
            calls.clear()
            for k,x in bd.local_rollout(model,self.sched,{},noise,self.knots,middle):x.square().mean().backward()
            self.assertEqual(len(calls),4)
            self.assertEqual([p.grad is not None for p in weights],[False,middle==2,middle==3,True])
            boundary,final=bd.rollout_boundary_and_final(model,self.sched,{},noise,self.knots,middle)
            reference=noise
            for i in range(middle):
                t=torch.full((2,),self.knots[i]);reference=self.sched.jump(reference,model(reference,t,{}),t,self.knots[i+1])
            torch.testing.assert_close(boundary,reference)
            self.assertFalse(boundary.requires_grad or final.requires_grad)

    def test_configuration_parses_and_rejects_mixed_protocols(self):
        from train import arguments
        from launch import flags
        cfg=json.loads((Path(__file__).parent/'configs/boundary_joint4_short.json').read_text())
        with patch('sys.argv',['train.py','--run','test',*flags(cfg)]):
            args=arguments();self.assertEqual(args.grad_clip,1);self.assertEqual(args.fake_updates,2)
        for extra in (['--variant','joint34'],['--pcm-boundary','919'],['--gan-view','tile']):
            with patch('sys.argv',['train.py','--run','test',*flags(cfg),*extra]),contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):arguments()

    def test_real_sam_architecture_feature_R1_and_frozen_input_gradients(self):
        from transformers import SamVisionConfig
        from transformers.models.sam.modeling_sam import SamVisionEncoder
        config=SamVisionConfig(hidden_size=32,num_hidden_layers=12,num_attention_heads=4,
            output_channels=16,image_size=64,patch_size=8,window_size=4,
            global_attn_indexes=[2,5,8,11],mlp_dim=64)
        disc=SAMDiscriminator(backbone=SamVisionEncoder(config))
        rgb=torch.randn(2,3,64,64,requires_grad=True)
        disc.set_heads(False)
        generator_loss(disc(rgb)).backward()
        self.assertGreater(rgb.grad.norm().item(),0)
        self.assertTrue(all(p.grad is None for p in disc.parameters()))
        disc.set_heads(True)
        parts=disc(rgb.detach(),return_parts=True,feature_r1=True)
        fake=disc(rgb.detach()+.2,return_parts=True)
        (discriminator_loss(parts['total'],fake['total'])+1e-5*parts['r1']).backward()
        self.assertTrue(torch.isfinite(parts['r1']))
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in disc.trainable()))
        self.assertTrue(all(p.grad is None for p in disc.backbone.parameters()))

    def test_whole_VAE_decode_keeps_gradient_and_does_not_crop(self):
        from diffusers import AutoencoderKL
        vae=AutoencoderKL(block_out_channels=(16,32),latent_channels=4,
            down_block_types=('DownEncoderBlock2D',)*2,up_block_types=('UpDecoderBlock2D',)*2,
            norm_num_groups=4,layers_per_block=1).eval().requires_grad_(False)
        latent=torch.randn(2,4,16,16,requires_grad=True)
        direct=vae.decode(latent/vae.config.scaling_factor,return_dict=False)[0]
        actual=decode_view(vae,latent,[(0,0)]*2,kind='whole',grad=True,chunk_size=1)
        self.assertEqual(actual.shape,(2,3,32,32))
        torch.testing.assert_close(actual,direct,atol=2e-5,rtol=2e-5)
        actual.square().mean().backward()
        self.assertGreater(latent.grad.norm().item(),0)
        self.assertTrue(all(p.grad is None for p in vae.parameters()))

    def test_SAM_loads_actual_transformers_safetensors_layout(self):
        from transformers import SamConfig,SamModel,SamVisionConfig
        v=SamVisionConfig(hidden_size=32,num_hidden_layers=12,num_attention_heads=4,
            output_channels=16,image_size=64,patch_size=8,window_size=4,
            global_attn_indexes=[2,5,8,11],mlp_dim=64)
        model=SamModel(SamConfig(vision_config=v.to_dict()))
        with tempfile.TemporaryDirectory() as folder:
            model.save_pretrained(folder,safe_serialization=True)
            loaded=SAMDiscriminator(folder)
            torch.testing.assert_close(loaded.backbone.patch_embed.projection.weight,
                                       model.vision_encoder.patch_embed.projection.weight)

    def test_protocol_metadata_roundtrips_for_resume(self):
        # In particular do not put integer-keyed phase dictionaries or tuples in
        # the resume-compared module metadata: JSON changes their Python types.
        from training_state import load_training
        from unittest.mock import patch
        modules=dict(objective='boundary_v1',ranges={str(k):list(v) for k,v in bd.phase_ranges(self.knots).items()})
        self.assertEqual(json.loads(json.dumps(modules)),modules)
        with tempfile.TemporaryDirectory() as folder:
            Path(folder,'complete.json').write_text(json.dumps({'world_size':1,'config':{'objective':'clean'}}))
            with patch('training_state.dist.get_world_size',return_value=1):
                with self.assertRaisesRegex(ValueError,'objective'):
                    load_training(folder,{}, {},{'objective':'boundary'})


if __name__=='__main__':unittest.main()
