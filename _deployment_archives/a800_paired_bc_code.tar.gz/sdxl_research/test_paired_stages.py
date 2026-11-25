import unittest,json,collections
from pathlib import Path
from unittest.mock import patch
import boundary as bd
from launch import flags
class PairedStagesTests(unittest.TestCase):
 def test_all_160_updates_match_student(self):
  for seed in [20261017,42]:
   expected=[]
   for g in range(160):
    pair=[bd.critic_stage(2*g+j,True,seed) for j in range(2)]
    self.assertEqual(pair,[bd.middle_stage(g,seed),4]);expected+=pair
   self.assertEqual(collections.Counter(expected)[4],160)
  self.assertEqual([bd.critic_stage(i) for i in range(6)],[2,3,4,2,3,4])
 def test_middle_uses_actual_noisy_boundary(self):
  import torch
  from diffusers import DDIMScheduler
  from objectives import VPSchedule
  torch.manual_seed(51)
  schedule=VPSchedule(DDIMScheduler(beta_start=.00085,beta_end=.012,beta_schedule='scaled_linear').alphas_cumprod)
  knots=[999,749,499,249,-1];noise=torch.randn(2,4,3,3)
  predict=lambda x,t,c:.1*x+.01
  for g in range(4):
   for update in [2*g,2*g+1]:
    k=bd.critic_stage(update,True,20261017)
    boundary,final=bd.rollout_boundary_and_final(predict,schedule,{},noise,knots,k)
    reference=noise
    for j in range(k):
     t=torch.full((len(noise),),knots[j],dtype=torch.long)
     reference=schedule.jump(reference,predict(reference,t,{}),t,knots[j+1])
    torch.testing.assert_close(boundary,reference)
    if k<4:self.assertFalse(torch.allclose(boundary,final))
    else:torch.testing.assert_close(boundary,final)
    t=torch.tensor([900,979]);eps=torch.randn_like(boundary)
    a_s,s_s=schedule.coefficients(knots[k],boundary);a_t,s_t=schedule.coefficients(t,boundary)
    expected=(a_t/a_s)*boundary+(s_t.square()-(a_t/a_s).square()*s_s.square()).sqrt()*eps
    torch.testing.assert_close(bd.renoise(schedule,boundary,knots[k],t,eps),expected)
    self.assertFalse(boundary.requires_grad)
 def test_real_configs_and_invalid_ratio(self):
  from train import arguments
  for arm in ['B','C']:
   c=json.loads((Path(__file__).parent/f'configs/turning_{arm}_quota_a800_g160.json').read_text())
   with patch('sys.argv',['train','--run','test',*flags(c)]):
    a=arguments();self.assertTrue(a.pair_fake_stages);self.assertTrue(a.match_fake_prior);self.assertEqual(a.fake_updates,2);self.assertEqual(a.variant,'dmd_only')
   import io,contextlib
   with patch('sys.argv',['train','--run','test',*flags(c),'--fake-updates','3']),contextlib.redirect_stderr(io.StringIO()):
    with self.assertRaises(SystemExit):arguments()
if __name__=='__main__':unittest.main()
