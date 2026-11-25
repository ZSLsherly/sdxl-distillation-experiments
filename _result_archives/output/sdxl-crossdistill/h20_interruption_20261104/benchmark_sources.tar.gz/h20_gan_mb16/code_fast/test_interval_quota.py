import unittest,json
from pathlib import Path
from turning_sampling import TurningSampler
class IntervalQuotaTests(unittest.TestCase):
 def test_exact_mass_and_stage_limits(self):
  root=Path(__file__).parent/'turning_profiles'
  samplers={a:TurningSampler(root/f'teacher_interval_quota_{a}_979.json',979) for a in ['B','C']}
  for stage,r in {2:(509,979),3:(259,979),4:(20,979)}.items():
   distributions={a:s.distribution(r) for a,s in samplers.items()}
   if stage<4:self.assertEqual(distributions['B'],distributions['C'])
   for arm,(ts,ps) in distributions.items():
    inside=lambda t:900<=t<=979 or (arm=='C' and 20<=t<=100)
    self.assertAlmostEqual(sum(p for t,p in zip(ts,ps) if inside(t)),.3,places=12)
    self.assertAlmostEqual(sum(p for t,p in zip(ts,ps) if not inside(t)),.7,places=12)
    for condition in [True,False]:
     values=[p for t,p in zip(ts,ps) if inside(t)==condition];self.assertEqual(min(values),max(values))
    draws=samplers[arm].sample(range(30000),20261017,time_range=r)
    self.assertTrue(all(r[0]<=t<=r[1] for t in draws))
    self.assertLess(abs(sum(inside(t) for t in draws)/len(draws)-.3),.012)
    if stage==4:
     self.assertEqual(ps,samplers[arm].profile['probabilities'])
 def test_configs_only_change_prior(self):
  root=Path(__file__).parent/'configs'
  b=json.loads((root/'turning_B_quota_a800_g160.json').read_text());c=json.loads((root/'turning_C_quota_a800_g160.json').read_text())
  self.assertNotEqual(b.pop('turning_profile'),c.pop('turning_profile'));self.assertEqual(b,c)
  self.assertEqual(b['checkpoint_mode'],'final-weights');self.assertEqual(b['save_every_updates'],0)
if __name__=='__main__':unittest.main()
