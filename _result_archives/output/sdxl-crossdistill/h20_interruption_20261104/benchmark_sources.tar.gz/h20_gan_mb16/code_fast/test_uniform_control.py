import json, unittest
from pathlib import Path
from unittest.mock import patch
import torch
from batching import random_inputs
from boundary import phase_ranges
from turning_sampling import TurningSampler
from launch import flags

class UniformControlTests(unittest.TestCase):
    def test_only_sampler_is_disabled(self):
        from train import arguments
        root=Path(__file__).parent/'configs'
        c=json.loads((root/'uniform_control_a800_g160.json').read_text())
        b=json.loads((root/'turning_C_quota_a800_g160.json').read_text())
        self.assertEqual(c,dict(b,turning_profile=None,match_fake_prior=False))
        with patch('sys.argv',['train','--run','test',*flags(c)]):
            a=arguments()
        self.assertIsNone(a.turning_profile)
        self.assertFalse(a.match_fake_prior)
        self.assertTrue(a.pair_fake_stages)
        self.assertEqual((a.batch,a.micro_batch,a.fake_updates),(64,16,2))

    def test_uniform_rng_and_other_inputs_preserved(self):
        torch.set_num_threads(1)
        sampler=TurningSampler(Path(__file__).parent/'turning_profiles/teacher_interval_quota_C_979.json',979)
        for k,r in phase_ranges([999,749,499,249,-1],10,979).items():
            for phase in ['student','fake']:
                pos=list(range(16))
                seed=20261017+1000*k if phase=='student' else 20261017
                plain=random_inputs(pos,phase,'cpu',seed=seed,shape=(4,48,48),pcm_boundary=749,time_range=r)
                prior=random_inputs(pos,phase,'cpu',seed=seed,shape=(4,48,48),pcm_boundary=749,time_range=r)
                sampler.apply(prior,pos,phase,seed,time_range=r,include_fake=True)
                for key in plain:
                    if key=='t':continue
                    if torch.is_tensor(plain[key]):self.assertTrue(torch.equal(plain[key],prior[key]),key)
                    else:self.assertEqual(plain[key],prior[key])
                # Independent replay of original torch stream: one latent then uniform integer.
                expected=[]
                for i in pos:
                    gen=torch.Generator().manual_seed(seed+(200000 if phase=='student' else 100000)+i)
                    torch.randn((4,48,48),generator=gen)
                    expected.append(torch.randint(r[0],r[1]+1,(),generator=gen).item())
                self.assertEqual(plain['t'].tolist(),expected)

if __name__=='__main__':unittest.main()
