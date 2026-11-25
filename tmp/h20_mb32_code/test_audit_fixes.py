"""Regression checks for the three independent-audit findings; CPU only."""
import contextlib
import io
import json
import random
from pathlib import Path
import tempfile
import unittest
from collections import Counter
from unittest.mock import patch

import torch

from batching import rollout_chunks
from objectives import VPSchedule, fake_stage, rollout_score_and_final
from evaluation_core import feature_blocks


class AuditFixTests(unittest.TestCase):
    def test_critic_cycles_all_stages_across_resume_boundary(self):
        self.assertEqual([fake_stage(i) for i in range(8)], [1,2,3,4]*2)
        continuous = [fake_stage(i) for i in range(800)]
        resumed = [fake_stage(i) for i in range(315)] + [fake_stage(i) for i in range(315,800)]
        self.assertEqual(continuous, resumed)
        self.assertEqual(Counter(continuous), {i:200 for i in (1,2,3,4)})

    def test_shared_trajectory_matches_separate_rollouts_and_fake_gradient(self):
        torch.manual_seed(17)
        schedule = VPSchedule(torch.linspace(.99,.01,1000))
        knots = [999,749,499,249,-1]
        noise = torch.randn(5,4,4,4,requires_grad=True)
        condition = {'bias': torch.randn(5,1,1,1)}
        calls=[]
        def predict(x,t,c):
            calls.append(int(t[0]))
            return .03*x + c['bias'] + t[:,None,None,None]/10000
        def separate(depth):
            x=noise.detach()
            for i in range(depth):
                t=torch.full((5,),knots[i],dtype=torch.long)
                x=schedule.jump(x,predict(x,t,condition),t,knots[i+1] if i<depth-1 else -1)
            return x
        final_reference=separate(4)
        for stage in (1,2,3,4):
            reference=separate(stage)
            calls.clear()
            score,final=rollout_chunks(
                lambda c,n: rollout_score_and_final(predict,schedule,c,n,knots,stage),
                condition,noise,chunk_size=2)
            self.assertEqual(calls,knots[:-1]*3)  # one trajectory per chunk, no duplicate prefix
            torch.testing.assert_close(score,reference,rtol=0,atol=0)
            torch.testing.assert_close(final,final_reference,rtol=0,atol=0)
            self.assertFalse(score.requires_grad)
            self.assertFalse(final.requires_grad)
            # The captured tensor, rather than the GAN final, drives fake fitting.
            weights=[]
            for x in (score,reference):
                w=torch.nn.Parameter(torch.tensor(.3))
                optimizer=torch.optim.SGD([w],lr=.01)
                (w*x-.2).square().mean().backward()
                optimizer.step();weights.append(w.detach())
            torch.testing.assert_close(*weights,rtol=0,atol=0)

    def test_double_student_offload_rejected_before_loading_models(self):
        from train import arguments
        with patch('sys.argv',['train.py','--run','test','--variant','joint4',
                              '--offload-student','--offload-student-optimizer']):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
                arguments()
        self.assertEqual(raised.exception.code,2)

    def test_old_final_only_checkpoint_cannot_silently_resume_new_protocol(self):
        from training_state import load_training
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)
            (path/'complete.json').write_text(json.dumps({'world_size':2,'config':{}}))
            with patch('training_state.dist.get_world_size',return_value=2):
                with self.assertRaisesRegex(ValueError,'fake_stage_policy'):
                    load_training(path,{}, {}, {'fake_stage_policy':'crossdistill_cycle_1_2_3_4_v1'})

    def test_resume_repacking_is_explicit_and_preserves_progress(self):
        from training_state import load_training
        old={'batch':64,'micro_batch':4,'rollout_batch':4}
        new={**old,'micro_batch':32,'rollout_batch':32}
        progress={'gcount':2,'fcount':10,'dcount':10}
        rng={'progress':progress,'python_rng':random.getstate(),'cpu_rng':torch.get_rng_state(),'cuda_rng':None}
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)
            (path/'complete.json').write_text(json.dumps({'world_size':2,'config':old}))
            with patch('training_state.dist.get_world_size',return_value=2),patch('training_state.dist.get_rank',return_value=0):
                with self.assertRaisesRegex(ValueError,'micro_batch'):
                    load_training(path,{}, {},new)
                with patch('training_state.torch.load',return_value=rng),patch('training_state.torch.cuda.set_rng_state'):
                    self.assertEqual(load_training(path,{}, {},new,allow_repacking=True),progress)
                with self.assertRaisesRegex(ValueError,'batch'):
                    load_training(path,{}, {},{**new,'batch':128},allow_repacking=True)

    def test_feature_cache_survives_worker_count_changes(self):
        for n in (1000,10000):
            files=set(feature_blocks(n,0,2))
            for world in (1,3,4,2):
                for rank in range(world):
                    files.update(feature_blocks(n,rank,world))
                ids=sorted(i for left,right in files for i in range(left,right))
                self.assertEqual(ids,list(range(n)))


if __name__=='__main__':
    unittest.main()
