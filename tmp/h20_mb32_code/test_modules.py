"""Numerical and routing checks for optional research modules; CPU only."""
import contextlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F
from stage_policy import make_stage_plan
from gan_losses import discriminator_loss, generator_loss
from image_views import decode_view, augment_view
from batching import decode_tiles
from objectives import discriminator_loss as old_discriminator_loss
from pixel import PixelDiscriminator


class LinearVAE(nn.Module):
    config=SimpleNamespace(scaling_factor=.5,block_out_channels=(1,1,1,1))
    def decode(self,x,return_dict=False):
        return (F.interpolate(x[:,:3],scale_factor=8,mode='nearest'),)


class ModuleTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        torch.set_num_threads(2)

    def test_default_stage_routes_match_all_existing_variants(self):
        for variant in ('joint4','joint34','gan4','dmd_only'):
            plan=make_stage_plan(variant)
            for g in range(12):
                stage=plan.student_stage(g)
                self.assertEqual(stage,2+g%3)
                self.assertEqual(stage in plan.dmd,not(variant=='gan4' and stage==4))
                self.assertEqual(stage in plan.gan,variant!='dmd_only' and (stage==4 or (variant=='joint34' and stage==3)))
            self.assertEqual([plan.critic_stage(f,7) for f in range(8)],[1,2,3,4]*2)
            self.assertEqual(json.loads(json.dumps(plan.metadata())),plan.metadata())

    def test_stage_replacements_and_invalid_combinations(self):
        for kind,expected in [('student_cycle',[2,3,4,2,3,4]),('match_student',[3]*6),('final',[4]*6)]:
            self.assertEqual([make_stage_plan('joint4',fake_policy=kind).critic_stage(f,1) for f in range(6)],expected)
        plan=make_stage_plan('joint4',student=[4],dmd=[],gan=[4])
        self.assertEqual(plan.student_stage(7),4)
        self.assertFalse(plan.dmd)
        for kwargs in ({'student':[1,2]}, {'student':[2,2]}, {'student':[]},
                       {'student':[4],'dmd':[2]}, {'dmd':[],'gan':[]}):
            with self.assertRaises(ValueError):make_stage_plan('joint4',**kwargs)

    def test_GAN_probe_counts_actual_updates_for_custom_cycles(self):
        plans=[make_stage_plan(v) for v in ('joint4','joint34','gan4','dmd_only')]
        plans.extend([make_stage_plan('joint4',student=[4],dmd=[],gan=[4]),
                      make_stage_plan('joint34',student=[4,2,3])])
        for plan in plans:
            seen=0
            for g in range(30):
                self.assertEqual(plan.gan_updates_before(g),seen)
                seen+=int(plan.student_stage(g) in plan.gan)
        single=plans[4]
        self.assertEqual([g for g in range(6) if single.gan_updates_before(g)%2==0],[0,2,4])
        joint=plans[1]
        self.assertEqual([g for g in range(9) if joint.student_stage(g) in joint.gan and joint.gan_updates_before(g)%2==0],[1,4,7])

    def test_hinge_values_and_gradients_preserve_default(self):
        real=torch.randn(5,requires_grad=True);fake=torch.randn(5,requires_grad=True)
        a=discriminator_loss(real,fake)
        b=old_discriminator_loss(real,fake)
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        for x,y in zip(torch.autograd.grad(a,(real,fake)),torch.autograd.grad(b,(real,fake))):
            torch.testing.assert_close(x,y,rtol=0,atol=0)
        torch.testing.assert_close(generator_loss(fake),-fake.mean(),rtol=0,atol=0)

    def test_logistic_has_correct_gradient_directions(self):
        real=torch.tensor([-100.,0,100.],requires_grad=True)
        fake=real.detach().clone().requires_grad_()
        dr,df=torch.autograd.grad(discriminator_loss(real,fake,'logistic'),(real,fake))
        dg=torch.autograd.grad(generator_loss(fake,'logistic'),fake)[0]
        self.assertTrue(torch.isfinite(dr).all() and torch.isfinite(df).all() and torch.isfinite(dg).all())
        self.assertTrue((dr<=0).all() and (df>=0).all() and (dg<=0).all())

    def test_tile_default_and_full_view_coordinates_gradients(self):
        vae=LinearVAE();z=torch.randn(2,4,64,64,requires_grad=True);crops=[(3,5),(8,11)]
        old=decode_tiles(vae,z,crops,grad=True,chunk_size=1)
        tile=decode_view(vae,z,crops,grad=True,chunk_size=1)
        full=decode_view(vae,z,crops,kind='full',grad=True)
        rgb=vae.decode(z/vae.config.scaling_factor)[0]
        expected=torch.cat([rgb[i:i+1,:,y*8+64:y*8+320,x*8+64:x*8+320] for i,(y,x) in enumerate(crops)])
        for x in (tile,full,expected):torch.testing.assert_close(x,old,rtol=0,atol=0)
        grads=[torch.autograd.grad(x.square().mean(),z,retain_graph=True)[0] for x in (old,tile,full,expected)]
        for g in grads[1:]:torch.testing.assert_close(g,grads[0],rtol=0,atol=0)
        # Real SDXL uses GroupNorm: equality of tile/full is not expected there.

    def test_augmentation_branches_preserve_input_gradients(self):
        x=torch.randn(2,3,4,4,requires_grad=True)
        flips=torch.tensor([True,False]).reshape(2,1,1,1)
        self.assertIs(augment_view(x,flips,'G'),x)
        self.assertIs(augment_view(x,flips,'D','none'),x)
        expected=torch.stack([x[0].flip(-1),x[1]])
        torch.testing.assert_close(augment_view(x,flips,'G','both'),expected)
        torch.testing.assert_close(augment_view(x,flips,'D'),expected)
        self.assertTrue(torch.isfinite(torch.autograd.grad(expected.square().sum(),x)[0]).all())

    def test_rgb_only_does_not_load_dino_and_keeps_G_gradient(self):
        with patch('pixel.Dinov2Model.from_pretrained',side_effect=AssertionError('must not load DINO')):
            disc=PixelDiscriminator('unused',kind='rgb')
        x=torch.randn(2,3,64,64,requires_grad=True);text=torch.randn(2,1280)
        disc.set_heads(True)
        discriminator_loss(disc(x,text),disc(x.detach()+.1,text)).backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in disc.trainable()))
        disc.zero_grad(set_to_none=True);disc.set_heads(False)
        parts=disc(x,text,return_parts=True)
        generator_loss(parts['total']).backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertTrue(all(p.grad is None for p in disc.trainable()))
        torch.testing.assert_close(parts['total'],parts['rgb'])
        self.assertTrue((parts['dino']==0).all())

    def test_short_with_resume_is_rejected(self):
        from launch import main
        config={'batch':64,'micro_batch':16,'resume':'example'}
        with patch('pathlib.Path.read_text',return_value=json.dumps(config)), patch('sys.argv',[
            'launch.py','--config','unused.json','--root','example','--gpus','0,1','--short']):
            with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit) as caught:main()
        self.assertEqual(caught.exception.code,2)


if __name__=='__main__':unittest.main()
