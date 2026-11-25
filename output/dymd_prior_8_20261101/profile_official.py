"""Read-only velocity instrumentation around the unmodified official PF-Wan engine."""
import json, math, os, runpy, time
from random import Random
from pathlib import Path
import torch
from project.engines import ENGINE_REGISTRY
from project.models.meta_model.wan2_1_i2v import Wan2_1_I2V

OUT = Path(os.environ['DYMD_PROFILE_DIR'])
OUT.mkdir(parents=True, exist_ok=True)
RANK = int(os.environ.get('RANK', 0))
PLAN = json.loads(Path(os.environ['DYMD_SAMPLE_PLAN']).read_text())
BY_FILE = {Path(r['filename']).name:r for r in PLAN}
orig_prepare = Wan2_1_I2V.prepare_inputs
orig_single = Wan2_1_I2V.pred_single
orig_step = Wan2_1_I2V.step_to

def directions(v):
    # C,T,H,W. Exclude the observed first temporal block, then temporal-center.
    v = v.detach().float().cpu()
    assert v.ndim == 4 and v.shape[1] == 21, v.shape
    future = v[:, 1:]
    fields = {'future_centered': future - future.mean(dim=1, keepdim=True),
              'future_uncentered': future, 'all_uncentered': v}
    result = {}
    for name, x in fields.items():
        x = x.flatten().double()
        norm = torch.linalg.vector_norm(x).item()
        assert math.isfinite(norm) and norm > 0, (name, norm)
        result[name] = (x / norm, norm)
    return result

def record(self, branch, v, sigma):
    if RANK != 0: return
    now = directions(v)
    for kind, (u, norm) in now.items():
        key = branch + '/' + kind
        row = {'sample': self._profile_index, 'step': self._profile_step,
               'branch': branch, 'centering': kind, 'sigma': sigma, 'norm': norm,
               'shape': list(v.shape)}
        if key in self._profile_prev:
            old_sigma, old_u = self._profile_prev[key]
            ds = abs(old_sigma - sigma)
            assert ds > 0
            cosine = torch.dot(old_u, u).clamp(-1, 1).item()
            angle = math.acos(cosine)
            row.update(sigma_previous=old_sigma, sigma_mid=(sigma+old_sigma)/2,
                       delta_sigma=ds, cosine=cosine, angle_rad=angle, kappa=angle/ds)
        self._profile_prev[key] = (sigma, u)
        with (OUT / ('trajectory_%02d.jsonl' % self._profile_index)).open('a') as f:
            f.write(json.dumps(row, allow_nan=False)+'\n')

def prepare(self, batch):
    entry = BY_FILE[Path(batch['filename']).name]
    self._profile_index = entry['sample']
    self.seed = entry['seed']
    self.random = Random(self.seed)
    self.generator = torch.Generator(device=self.device).manual_seed(self.seed)
    self._profile_step = 0
    self._profile_prev = {}
    if RANK == 0:
        dest = OUT / ('trajectory_%02d.jsonl' % self._profile_index)
        if dest.exists(): raise FileExistsError(f'Refusing to mix runs: {dest}')
        meta = dict(batch, sample=self._profile_index, seed=self.seed, condition=entry['condition'])
        (OUT / ('input_%02d.json' % self._profile_index)).write_text(json.dumps(meta,indent=2))
        print('PROFILE_SAMPLE_START',self._profile_index,flush=True)
    return orig_prepare(self, batch)

def single(self, model, inputs):
    pred = orig_single(self, model, inputs)
    # Official pred_cfg batches positive followed by negative, without extra evaluations.
    if RANK == 0:
        sigma = float(inputs['t'].max().item()) / self.schedule.T
        assert len(pred) == 2, len(pred)
        record(self, 'conditional', pred[0], sigma)
        record(self, 'negative', pred[1], sigma)
    return pred

def step(self, pred, inputs, s):
    sigma = float(inputs['t'].max().item()) / self.schedule.T
    record(self, 'cfg5', pred[0], sigma)
    result = orig_step(self, pred, inputs, s)
    self._profile_step += 1
    if RANK == 0:
        print('PROFILE_STEP',self._profile_index,self._profile_step,'sigma',sigma,flush=True)
        if self._profile_step == 40:
            (OUT / ('trajectory_%02d.complete' % self._profile_index)).write_text(str(time.time()))
            self._profile_prev.clear()
        progress={'target':len(PLAN),'sample':self._profile_index,'step':self._profile_step,
                  'completed':len(list(OUT.glob('trajectory_*.complete'))),'updated_unix':time.time()}
        temp=OUT.parent/'progress.tmp'
        temp.write_text(json.dumps(progress));temp.replace(OUT.parent/'progress.json')
    return result

Wan2_1_I2V.prepare_inputs = prepare
Wan2_1_I2V.pred_single = single
Wan2_1_I2V.step_to = step
runpy.run_path('tools/main.py', run_name='__main__')
