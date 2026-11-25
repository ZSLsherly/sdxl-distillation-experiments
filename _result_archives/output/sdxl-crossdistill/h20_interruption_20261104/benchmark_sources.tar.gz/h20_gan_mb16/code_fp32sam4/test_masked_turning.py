import unittest,json,math
from pathlib import Path
from turning_sampling import TurningSampler
class MaskedPriorTests(unittest.TestCase):
 def test_mask_and_stage_intersection(self):
  samplers={a:TurningSampler(Path(__file__).parent/f'turning_profiles/normalized_u_{a}_masked_979.json',979) for a in ['B','C']}
  for stage,r in {2:(509,979),3:(259,979),4:(20,979)}.items():
   distributions={a:s.distribution(r) for a,s in samplers.items()}
   if stage<4:self.assertEqual(distributions['B'],distributions['C'])
   for a,(ts,ps) in distributions.items():
    self.assertAlmostEqual(sum(ps),1,places=12)
    self.assertTrue(all(math.isfinite(x) and x>0 for x in ps))
    for t,p in zip(ts,ps):
     inside=900<=t<=979 or (a=='C' and 20<=t<=100)
     if not inside:self.assertAlmostEqual(p,.7/len(ts),places=15)
    draws=samplers[a].sample(range(20000),20261017,time_range=r)
    expected=sum(p for t,p in zip(ts,ps) if t>=900)
    self.assertLess(abs(sum(t>=900 for t in draws)/len(draws)-expected),.015)
    self.assertTrue(all(r[0]<=t<=r[1] for t in draws))
 def test_configs_differ_only_in_prior(self):
  root=Path(__file__).parent
  b=json.loads((root/'configs/turning_B_masked_a800_g160.json').read_text())
  c=json.loads((root/'configs/turning_C_masked_a800_g160.json').read_text())
  self.assertNotEqual(b.pop('turning_profile'),c.pop('turning_profile'))
  self.assertEqual(b,c)
  self.assertEqual(b['checkpoint_mode'],'final-weights')
  self.assertEqual(b['save_every_updates'],0)
  self.assertFalse(b['benchmark'])
 def test_empty_intersection_is_uniform(self):
  s=TurningSampler(Path(__file__).parent/'turning_profiles/normalized_u_B_masked_979.json',979)
  t,p=s.distribution((200,300));self.assertEqual(p,[1/101]*101)
if __name__=='__main__':unittest.main()
