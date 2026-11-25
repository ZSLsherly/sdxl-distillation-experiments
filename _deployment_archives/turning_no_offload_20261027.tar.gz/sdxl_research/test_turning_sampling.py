import json,unittest
from pathlib import Path
import torch
from turning_sampling import TurningSampler
from batching import random_inputs

ROOT=Path(__file__).parent
class TurningAblationTests(unittest.TestCase):
    def setUp(self):
        self.samplers=[TurningSampler(ROOT/'turning_profiles'/f'{n}_512.json') for n in ['normalized_u','raw_angle']]

    def test_only_student_re_noise_time_changes(self):
        for phase in ['student','fake']:
            ids=list(range(8));old=random_inputs(ids,phase,'cpu',shape=(4,64,64),pcm_boundary=749)
            for sampler in self.samplers:
                new=random_inputs(ids,phase,'cpu',shape=(4,64,64),pcm_boundary=749)
                sampler.apply(new,ids,phase,20261017)
                for k,v in old.items():
                    if phase=='student' and k=='t':continue
                    if torch.is_tensor(v):self.assertTrue(torch.equal(v,new[k]),(phase,k))
                    else:self.assertEqual(v,new[k])
                if phase=='student':self.assertFalse(torch.equal(old['t'],new['t']))

    def test_packing_and_rank_invariant(self):
        for s in self.samplers:
            expected=s.sample(range(128),20261017)
            for world in [1,2,4]:
                result={}
                for rank in range(world):
                    ids=list(range(rank,128,world))
                    for start in range(0,len(ids),7):
                        chunk=ids[start:start+7];result.update(zip(chunk,s.sample(chunk,20261017)))
                self.assertEqual(expected,[result[i] for i in range(128)])

    def test_empirical_probabilities(self):
        for s in self.samplers:
            p=json.loads(s.path.read_text());draws=s.sample(range(40000),20261017)
            self.assertTrue(all(20<=t<=919 for t in draws))
            for lo,hi in [(20,199),(200,799),(800,919)]:
                expected=sum(v for t,v in zip(p['timesteps'],p['probabilities']) if lo<=t<=hi)
                actual=sum(lo<=t<=hi for t in draws)/len(draws)
                self.assertLess(abs(expected-actual),.01)

    def test_matched_configuration_and_source(self):
        configs=[json.loads((ROOT/'configs'/f'turning_{n}_g160.json').read_text()) for n in ['normalized_u','raw_angle']]
        for c in configs:c.pop('turning_profile')
        self.assertEqual(*configs)
        self.assertEqual(self.samplers[0].metadata['source_sha256'],self.samplers[1].metadata['source_sha256'])
        self.assertNotEqual(self.samplers[0].metadata['distribution_sha256'],self.samplers[1].metadata['distribution_sha256'])

    def test_rng_state_unchanged(self):
        import random
        py=random.getstate();cpu=torch.get_rng_state().clone()
        self.samplers[0].sample(range(64),5)
        self.assertEqual(py,random.getstate());self.assertTrue(torch.equal(cpu,torch.get_rng_state()))

    def test_boundary_support_and_fixed_mixture(self):
        for s in self.samplers:
            for lo,hi in [(509,739),(259,489),(20,239)]:
                ts,ps=s.distribution((lo,hi));self.assertAlmostEqual(sum(ps),1.)
                self.assertGreaterEqual(min(ps),.7/len(ts))
                draws=s.sample(range(20000),20261017+1000*lo,(lo,hi))
                self.assertTrue(all(lo<=t<=hi for t in draws))
                split=(lo+hi)/2
                self.assertLess(abs(sum(t>split for t in draws)/len(draws)-sum(p for t,p in zip(ts,ps) if t>split)),.015)

    def test_boundary_other_random_inputs_unchanged(self):
        for stage,bounds in [(2,(509,739)),(3,(259,489)),(4,(20,239))]:
            positions=list(range(8));seed=20261017+1000*stage
            old=random_inputs(positions,'student','cpu',shape=(4,64,64),seed=seed,pcm_boundary=749,time_range=bounds)
            for sampler in self.samplers:
                new=random_inputs(positions,'student','cpu',shape=(4,64,64),seed=seed,pcm_boundary=749,time_range=bounds)
                sampler.apply(new,positions,'student',seed,time_range=bounds)
                for key,v in old.items():
                    if key=='t':continue
                    if torch.is_tensor(v):self.assertTrue(torch.equal(v,new[key]),key)
                    else:self.assertEqual(v,new[key])

if __name__=='__main__':unittest.main()
