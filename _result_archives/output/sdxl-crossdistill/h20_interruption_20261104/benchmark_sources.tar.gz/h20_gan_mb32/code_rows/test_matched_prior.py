import unittest,torch,json
from pathlib import Path
from turning_sampling import TurningSampler
from batching import random_inputs
class MatchedPriorTests(unittest.TestCase):
 def test_fake_changes_only_time(self):
  root=Path(__file__).parent/'turning_profiles'
  for arm in ['B','C']:
   s=TurningSampler(root/f'teacher_interval_quota_{arm}_979.json',979)
   for k,r in {2:(509,979),3:(259,979),4:(20,979)}.items():
    ids=list(range(8));before=random_inputs(ids,'fake','cpu',shape=(4,48,48),time_range=r)
    after=random_inputs(ids,'fake','cpu',shape=(4,48,48),time_range=r)
    s.apply(after,ids,'fake',20261017+500000+1000*k,time_range=r,include_fake=True)
    self.assertEqual(after['t'].tolist(),s.sample(ids,20261017+500000+1000*k,time_range=r))
    for key in before:
     if key=='t':continue
     if torch.is_tensor(before[key]):self.assertTrue(torch.equal(before[key],after[key]),key)
     else:self.assertEqual(before[key],after[key])
 def test_same_law_independent_streams_and_packing(self):
  root=Path(__file__).parent/'turning_profiles'
  for arm in ['B','C']:
   s=TurningSampler(root/f'teacher_interval_quota_{arm}_979.json',979)
   for k,r in {2:(509,979),3:(259,979),4:(20,979)}.items():
    ids=list(range(20000));g=s.sample(ids,20261017+1000*k,time_range=r);f=s.sample(ids,20261017+500000+1000*k,time_range=r)
    self.assertNotEqual(g,f)
    inside=lambda t:900<=t<=979 or (arm=='C' and 20<=t<=100)
    for draws in [g,f]:
     self.assertTrue(all(r[0]<=t<=r[1] for t in draws))
     self.assertLess(abs(sum(inside(t) for t in draws)/len(draws)-.3),.015)
    split=sum([s.sample(ids[i:i+16],20261017+500000+1000*k,time_range=r) for i in range(0,64,16)],[])
    self.assertEqual(split,f[:64])
 def test_BC_both_enable_matching(self):
  root=Path(__file__).parent/'configs'
  for arm in ['B','C']:
   c=json.loads((root/f'turning_{arm}_quota_a800_g160.json').read_text());self.assertTrue(c['match_fake_prior'])
if __name__=='__main__':unittest.main()
