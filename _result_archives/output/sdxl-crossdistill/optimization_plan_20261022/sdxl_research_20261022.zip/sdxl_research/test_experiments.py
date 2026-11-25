import contextlib,copy,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from runtime import schedule_settings
from launch import command, gpu_check
from budget import plan
from feature_metrics import kid

class ExperimentTests(unittest.TestCase):
    def test_launch_rejects_busy_or_reset_selected_gpus(self):
        healthy='2, GPU-test, 48000, 0, P8\n'
        with patch('launch.subprocess.check_output',side_effect=[healthy,'']):
            self.assertEqual(len(gpu_check([2],36)),1)
        with patch('launch.subprocess.check_output',side_effect=[healthy,'GPU-test, 123\n']):
            with self.assertRaises(RuntimeError):gpu_check([2],36)
        with patch('launch.subprocess.check_output',side_effect=['2, GPU-test, 48000, [N/A], [GPU requires reset]\n','']):
            with self.assertRaises(RuntimeError):gpu_check([2],36)

    def test_configs_keep_controlled_factors(self):
        root=Path(__file__).parent/'configs'
        for schedule in ('legacy','uniform'):
            d=json.loads((root/f'{schedule}_dmd.json').read_text())
            g=json.loads((root/f'{schedule}_joint4.json').read_text())
            self.assertEqual({k:v for k,v in d.items() if k!='variant'}, {k:v for k,v in g.items() if k!='variant'})
            for world in (2,4):
                cmd=command(g,Path('/example'),'test',world)
                self.assertIn('--batch',cmd);self.assertEqual(cmd[cmd.index('--batch')+1],'64')
        for p in root.rglob('*.json'):
            if p.name=='models.example.json':continue
            cfg=json.loads(p.read_text());schedule_settings(cfg['schedule'],cfg.get('dmd_t_max'),cfg.get('pcm_boundary'))

    def test_budget_does_not_exceed_total(self):
        for seconds in (119,285):
            out=plan(seconds,72,2,12,64)
            self.assertLessEqual(out['estimated_training_hours_total']+12,72)
            self.assertEqual(out['max_G_updates_per_run']%3,0)

    def test_kid_matches_explicit_unbiased_estimator(self):
        x=np.arange(12).reshape(4,3)/10;y=x+0.2
        def kernel(a,b):return (a.dot(b)/3+1)**3
        expected=(sum(kernel(a,b) for i,a in enumerate(x) for j,b in enumerate(x) if i!=j)+
                  sum(kernel(a,b) for i,a in enumerate(y) for j,b in enumerate(y) if i!=j))/12
        expected-=2*sum(kernel(a,b) for a in x for b in y)/16
        self.assertAlmostEqual(kid(x,y,subset=4,repeats=1)['KID_raw'],expected,places=12)

    def test_evaluation_uses_checkpoint_schedule(self):
        import evaluation_core as core
        original={name:getattr(torch,name) for name in ('randn','full','tensor','Generator')}
        def cpu(name):
            def call(*args,**kwargs):
                if kwargs.get('device')=='cuda':kwargs['device']='cpu'
                return original[name](*args,**kwargs)
            return call
        times=[]
        class Pipe:
            _research_knots=[999,749,499,249,-1]
            vae=SimpleNamespace(config=SimpleNamespace(scaling_factor=1),decode=lambda x,**kw:(x[:,:3],))
            def encode_prompt(self,prompts,**kwargs):return (torch.zeros(1,1,2),None,torch.zeros(1,2),None)
            def unet(self,x,t,**kwargs):times.append(int(t[0]));return (torch.zeros_like(x),)
        with contextlib.ExitStack() as stack:
            for name in original:stack.enter_context(patch.object(torch,name,cpu(name)))
            stack.enter_context(patch.object(torch,'autocast',lambda *a,**kw:contextlib.nullcontext()))
            stack.enter_context(patch.object(core,'RUNS',{'custom':{}}))
            stack.enter_context(patch.object(core.DDIMScheduler,'from_pretrained',return_value=SimpleNamespace(alphas_cumprod=torch.linspace(.99,.01,1000))))
            images,_=core.generate(Pipe(),'custom',['test'],[123])
        self.assertEqual(times,[999,749,499,249]);self.assertEqual(len(images),1)

if __name__=='__main__':unittest.main()
