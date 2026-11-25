import copy
import importlib.util
from pathlib import Path
import sys
import unittest
import torch
from runtime import average_gradients, schedule_settings
from batching import random_inputs

class RuntimeTests(unittest.TestCase):
    def test_bucketed_mean_and_optimizer_update(self):
        torch.manual_seed(3)
        a=torch.nn.Sequential(torch.nn.Linear(7,13),torch.nn.Linear(13,2))
        b=copy.deepcopy(a)
        # Model two ranks with known peer gradients, including an unused parameter.
        for p,q in zip(a.parameters(),b.parameters()):
            p.grad=torch.randn_like(p);q.grad=p.grad.clone()
        seen=[]
        def reduce(value):
            seen.append(value.numel());value.mul_(3)
        calls=average_gradients(a.parameters(),16,world_size=2,reduce=reduce)
        self.assertEqual(calls,1)
        average_gradients(b.parameters(),0,world_size=2,reduce=reduce)
        oa=torch.optim.AdamW(a.parameters(),lr=1e-6)
        ob=torch.optim.AdamW(b.parameters(),lr=1e-6)
        for p,q in zip(a.parameters(),b.parameters()):
            torch.testing.assert_close(p.grad,q.grad,rtol=0,atol=0)
        oa.step();ob.step()
        for p,q in zip(a.parameters(),b.parameters()):
            torch.testing.assert_close(p,q,rtol=0,atol=0)

    def test_small_buckets_mixed_dtype_and_none(self):
        ps=[torch.nn.Parameter(torch.zeros(8,dtype=t)) for t in (torch.float32,torch.float64,torch.float32)]
        for p in ps[:2]:p.grad=torch.ones_like(p)
        self.assertEqual(average_gradients(ps,.000001,2,lambda g:g.mul_(2)),2)
        self.assertIsNone(ps[2].grad)
        for p in ps[:2]:self.assertTrue(torch.equal(p.grad,torch.ones_like(p)))

    def test_sampling_boundaries_are_independent(self):
        self.assertEqual(schedule_settings([999,749,499,249]),([999,749,499,249,-1],749,919))
        self.assertEqual(schedule_settings([999,749,499,249],749),([999,749,499,249,-1],749,749))
        self.assertEqual(schedule_settings([999,749,499,249],919,919),([999,749,499,249,-1],919,919))
        for bad in ([999,999,499,199],[998,749,499,249],[999,749,499,-1]):
            with self.assertRaises(ValueError):schedule_settings(bad)

    def test_inputs_match_baseline_except_intended_times(self):
        spec=importlib.util.spec_from_file_location('baseline_batching',Path(__file__).resolve().parent.parent/'sdxl_parallel/batching.py')
        old=importlib.util.module_from_spec(spec);sys.modules[spec.name]=old;spec.loader.exec_module(old)
        for phase in ('student','fake'):
            kwargs=dict(seed=20261017,shape=(4,48,48))
            expected=old.random_inputs(range(64),phase,'cpu',**kwargs)
            actual=random_inputs(range(64),phase,'cpu',**kwargs)
            changed=random_inputs(range(64),phase,'cpu',pcm_boundary=749,dmd_t_max=749,**kwargs)
            for k in expected:
                if torch.is_tensor(expected[k]):
                    self.assertTrue(torch.equal(expected[k],actual[k]),k)
                    if phase=='fake' or k not in ('t','pcm_t'):
                        self.assertTrue(torch.equal(expected[k],changed[k]),k)
                else:self.assertEqual(expected[k],changed[k])
            if phase=='student':
                self.assertTrue(((changed['t']>=20)&(changed['t']<=749)).all())
                self.assertTrue(((changed['pcm_t']>=770)&(changed['pcm_t']<=999)).all())

if __name__=='__main__':unittest.main()
