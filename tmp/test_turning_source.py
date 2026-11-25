import json
import unittest
from pathlib import Path
from turning_sampling import TurningSampler

class TurningTests(unittest.TestCase):
    def setUp(self):
        self.path = Path(__file__).with_name("turning_distribution.json")
        self.s = TurningSampler(self.path)

    def test_global_positions_are_packing_and_rank_invariant(self):
        expected = self.s.sample(range(128),20261017)
        for world in [1,2,4]:
            for micro in [1,4,8,16]:
                actual = {}
                for rank in range(world):
                    ids = list(range(rank,128,world))
                    for i in range(0,len(ids),micro):
                        chunk=ids[i:i+micro]
                        actual.update(zip(chunk,self.s.sample(chunk,20261017)))
                self.assertEqual(expected,[actual[i] for i in range(128)])

    def test_distribution_and_empirical_mass(self):
        p=json.loads(self.path.read_text())
        draws=self.s.sample(range(60000),20261017)
        self.assertTrue(all(20<=t<=919 for t in draws))
        self.assertLess(abs(sum(draws)/len(draws)-p["mean_t"]),4.)
        for low in range(20,920,100):
            expected=sum(q for t,q in zip(p["timesteps"],p["probabilities"]) if low<=t<low+100)
            actual=sum(low<=t<low+100 for t in draws)/len(draws)
            self.assertLess(abs(actual-expected),.008)

    def test_other_random_inputs_unchanged(self):
        import torch
        from batching import random_inputs
        positions=list(range(8))
        before=random_inputs(positions,"student","cpu",shape=(4,64,64))
        after=random_inputs(positions,"student","cpu",shape=(4,64,64))
        after["t"]=torch.tensor(self.s.sample(positions,20261017))
        for name in before:
            if name=="t": continue
            if torch.is_tensor(before[name]):
                self.assertTrue(torch.equal(before[name],after[name]),name)
            else:
                self.assertEqual(before[name],after[name],name)
        self.assertFalse(torch.equal(before["t"],after["t"]))

if __name__=="__main__":
    unittest.main()
