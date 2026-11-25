"""CPU checks for the changed stage boundary and unchanged sample streams."""
import importlib.util
from pathlib import Path
import sys
import unittest
import torch
from batching import random_inputs

spec = importlib.util.spec_from_file_location('baseline_batching', Path(__file__).resolve().parent.parent/'sdxl_parallel/batching.py')
baseline = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = baseline
spec.loader.exec_module(baseline)

class ScheduleTests(unittest.TestCase):
    def test_ranges_and_original_random_streams(self):
        for phase in ('fake', 'student'):
            positions = list(range(64))
            args = dict(seed=20261017, shape=(4,48,48))
            old = baseline.random_inputs(positions, phase, 'cpu', **args)
            compat = random_inputs(positions, phase, 'cpu', **args)
            new = random_inputs(positions, phase, 'cpu', boundary=749, **args)
            for key in old:
                if torch.is_tensor(old[key]):
                    self.assertTrue(torch.equal(old[key], compat[key]),key)
                    if phase == 'fake' or key not in ('t','pcm_t'):
                        self.assertTrue(torch.equal(old[key],new[key]),key)
                else:
                    self.assertEqual(old[key],compat[key]);self.assertEqual(old[key],new[key])
            if phase=='student':
                self.assertTrue(((new['t']>=20)&(new['t']<=749)).all())
                self.assertTrue(((new['pcm_t']>=770)&(new['pcm_t']<=999)).all())
                self.assertTrue((new['pcm_t']-20>749).all())
            # Concatenated per-image execution must match the whole batch.
            for i in (0,15,31,63):
                single=random_inputs([i],phase,'cpu',boundary=749,**args)
                for key in ('noise','t','eps'):
                    self.assertTrue(torch.equal(single[key][0],new[key][i]),key)

if __name__=='__main__':
    unittest.main()
