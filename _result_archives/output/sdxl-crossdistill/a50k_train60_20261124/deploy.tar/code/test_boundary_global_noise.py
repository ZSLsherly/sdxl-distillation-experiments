import unittest
import torch
from diffusers import UNet2DConditionModel, DDIMScheduler
import boundary as bd
from objectives import VPSchedule

class GlobalNoiseTests(unittest.TestCase):
    def test_global_ranges(self):
        self.assertEqual(bd.phase_ranges([999,749,499,249,-1],10,979),{2:(509,979),3:(259,979),4:(20,979)})
    def test_high_noise_kernel_and_gradient(self):
        schedule=VPSchedule(DDIMScheduler(beta_start=.00085,beta_end=.012,beta_schedule='scaled_linear').alphas_cumprod)
        for boundary in (499,249,-1):
            y=torch.randn(2,4,4,4,requires_grad=True);t=torch.tensor([900,979]);eps=torch.randn_like(y)
            a,b,_=bd.coefficients(schedule,boundary,t,y)
            aa,ss=schedule.coefficients(boundary,y);at,st=schedule.coefficients(t,y)
            torch.testing.assert_close(a*aa,at)
            torch.testing.assert_close(a*a*ss*ss+b*b,st*st)
            xt=bd.renoise(schedule,y.detach(),boundary,t,eps)
            fake=torch.randn_like(y,requires_grad=True);teacher=torch.randn_like(y,requires_grad=True)
            loss,direction=bd.dm_surrogate(schedule,y,boundary,t,xt,fake,teacher)
            loss.backward()
            torch.testing.assert_close(y.grad,direction/y.numel())
            self.assertIsNone(fake.grad);self.assertIsNone(teacher.grad)
            self.assertTrue(torch.isfinite(y.grad).all())
    def test_actual_unet_stage_embedding(self):
        torch.set_num_threads(2)
        model=UNet2DConditionModel(sample_size=8,in_channels=4,out_channels=4,
            down_block_types=('DownBlock2D',),up_block_types=('UpBlock2D',),
            block_out_channels=(32,),layers_per_block=1,cross_attention_dim=32,norm_num_groups=8)
        x=torch.randn(2,4,8,8);text=torch.randn(2,4,32)
        with torch.no_grad():reference=model(x,900,text).sample
        bd.add_fake_stage_condition(model)
        for k in (2,3,4):
            torch.testing.assert_close(model(x,900,text,class_labels=bd.stage_labels(k,x)).sample,reference,rtol=0,atol=0)
        model(x,900,text,class_labels=bd.stage_labels(3,x)).sample.square().mean().backward()
        grad=model.class_embedding.weight.grad
        self.assertGreater(grad[1].abs().sum().item(),0)
        self.assertEqual(grad[[0,2]].abs().sum().item(),0)
        # Strict checkpoint roundtrip includes new stage parameters.
        clone=UNet2DConditionModel.from_config(model.config)
        clone.load_state_dict(model.state_dict(),strict=True)
        torch.testing.assert_close(clone(x,900,text,class_labels=bd.stage_labels(3,x)).sample,
                                   model(x,900,text,class_labels=bd.stage_labels(3,x)).sample)

if __name__=='__main__':unittest.main()
