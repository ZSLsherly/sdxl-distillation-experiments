import unittest
import torch
from torch import nn
from fake_tracking import mix_shared_parameters, terminal_probe
from objectives import VPSchedule


class TrackingTests(unittest.TestCase):
    def test_direction_and_fake_only_embedding(self):
        student=nn.Linear(3,2)
        fake=nn.Linear(3,2)
        fake.class_embedding=nn.Embedding(3,2)
        original={k:v.detach().clone() for k,v in fake.named_parameters()}
        src={k:v.detach().clone() for k,v in student.named_parameters()}
        mix_shared_parameters(fake,student,.01)
        for k,v in student.named_parameters():
            torch.testing.assert_close(src[k],v)
            torch.testing.assert_close(dict(fake.named_parameters())[k],.99*original[k]+.01*src[k])
        torch.testing.assert_close(fake.class_embedding.weight,original['class_embedding.weight'])

    def test_counterfactual_same_inputs(self):
        schedule=VPSchedule(torch.linspace(.99,.05,1000))
        clean=torch.randn(8,4,3,3)
        seen=[]
        value=[.15]
        def predict(x,t):
            seen.append(x.clone())
            return value[0]*x
        rows=terminal_probe(schedule,clean,predict,lambda x,t:.4*x,lambda:value.__setitem__(0,.3),38)
        self.assertEqual([r['time'] for r in rows],[60,499,940])
        for i,r in enumerate(rows):
            torch.testing.assert_close(seen[i],seen[i+3])
            self.assertNotEqual(r['before'],r['after'])
            self.assertGreater(r['relative_direction_change'],0.)
        rows=terminal_probe(schedule,clean,predict,lambda x,t:.4*x,lambda:None,38)
        self.assertTrue(all(r['before']==r['after'] for r in rows))
