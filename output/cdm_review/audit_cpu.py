"""Small CPU checks of unmodified CDM functions, without loading any diffusion model."""
import ast
import json
import sys
from pathlib import Path
import torch
import numpy as np
from PIL import Image

torch.set_num_threads(2)
torch.manual_seed(72)
if "sources" not in globals():
    root = Path(__file__).resolve().parents[2] / "research_repos/CDM"
    sources = {name: (root/name).read_text(encoding="utf8") for name in [
        "cdm/training/losses.py", "cdm/utils/common.py",
        "cdm/pipeline/pipeline.py", "cdm/rewards/fid_scorer.py"]}

def extract(file, names, namespace):
    tree = ast.parse(sources[file])
    tree.body = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    exec(compile(tree, file, "exec"), namespace)
    return namespace

from typing import Optional, List
ns = {"torch": torch, "Optional": Optional, "List": List}
extract("cdm/training/losses.py",
        {"compute_cfg_loss", "compute_dm_loss", "compute_fake_teacher_diffusion_loss"}, ns)
extract("cdm/utils/common.py", {"sample_sigmas"}, ns)
result = {"device": "CPU only", "torch_version": torch.__version__, "checks": {}}
checks = result["checks"]
shape = (3, 2, 4, 4)
x, z = torch.randn(shape), torch.randn(shape)
t = torch.tensor([.2, .5, .9]).view(-1, 1, 1, 1)
tau = torch.tensor([.3, .6, .8]).view(-1, 1, 1, 1)
for kind, fname in [("DM", "compute_dm_loss"), ("CA", "compute_cfg_loss")]:
    v = torch.randn(shape, requires_grad=True)
    real = torch.randn(shape, requires_grad=True)
    other = torch.randn(shape, requires_grad=True)
    common = dict(student_prediction=v, xt=x, xt_teacher=z,
                  student_sigma_expanded=t, teacher_sigma_expanded=tau)
    if kind == "DM":
        loss, _ = ns[fname](**common, real_teacher_prediction=real, fake_teacher_prediction=other)
        scale = 1
    else:
        loss, _ = ns[fname](**common, real_teacher_cond_prediction=real,
                           real_teacher_uncond_prediction=other, guidance_scale=7)
        scale = 7
    loss.backward()
    d = x-t*v.detach()
    dr, df = z-tau*real.detach(), z-tau*other.detach()
    delta = scale*(dr-df)/((dr-d).abs().mean((1,2,3), keepdim=True)+1e-8).clamp_min(.1)
    expected = 2*t*delta/v.numel()
    checks[kind] = {"gradient_max_abs_error": float((v.grad-expected).abs().max()),
                    "teacher_grad_is_none": real.grad is None,
                    "other_teacher_grad_is_none": other.grad is None,
                    "gradient_multiplier_vs_half_mean_MSE": 2}
    assert torch.allclose(v.grad, expected, atol=2e-6, rtol=2e-5)
    assert real.grad is None and other.grad is None

v = torch.randn(shape, requires_grad=True)
clean = torch.randn(shape, requires_grad=True)
eps = torch.randn(shape)
loss, _ = ns["compute_fake_teacher_diffusion_loss"](v, clean, eps, t)
loss.backward()
checks["fake_flow_matching"] = {"student_target_grad_is_none": clean.grad is None,
                              "gradient_max_abs_error": float((v.grad-2*(v.detach()-eps+clean.detach())/v.numel()).abs().max())}
assert clean.grad is None

torch.manual_seed(17)
uncapped = ns["sample_sigmas"](10000, "uniform", "cpu", upper_bound=torch.full((10000,), .3))
torch.manual_seed(17)
capped = ns["sample_sigmas"](10000, "uniform_capped", "cpu", upper_bound=torch.full((10000,), .3))
checks["time_sampling"] = {
    "uniform_fraction_above_supplied_upper_bound_0_3": float((uncapped>.3).float().mean()),
    "capped_max": float(capped.max()),
    "uniform_boundary_mass_0_02": float((uncapped==.02).float().mean()),
    "uniform_boundary_mass_0_98": float((uncapped==.98).float().mean())}
assert uncapped.max()>.3 and capped.max()<=.3

extract("cdm/pipeline/pipeline.py", {"TrainingSchedulerWrapper"}, ns)
from types import SimpleNamespace
class Scheduler:
    def __init__(self):
        self._step_index=0
        self.sigmas=torch.tensor([1., .5, 0.])
        self.config=SimpleNamespace(stochastic_sampling=False)
    def step(self,v,t,x,**kwargs):
        i=self._step_index
        self._step_index+=1
        return (x+(self.sigmas[i+1]-self.sigmas[i])*v,)
s=Scheduler();w=ns["TrainingSchedulerWrapper"](s);w.install()
initial=torch.ones(2,1,2,2)
with torch.no_grad():
    x1=s.step(torch.full_like(initial,.25),1000,initial)[0]
    s.step(torch.full_like(initial,.5),500,x1)
xt,x0,vel=w.collect();w.uninstall()
checks["trajectory_indexing"] = {"cached_x0_index_i_plus_1_matches_anchor_i": all(
    torch.allclose(x0[i+1],xt[i]-s.sigmas[i]*vel[i]) for i in range(2)),
    "cached_velocity_requires_grad": [v.requires_grad for v in vel]}
assert checks["trajectory_indexing"]["cached_x0_index_i_plus_1_matches_anchor_i"]

# Inspect the actual scorer's generated-image preprocessing, replacing only Inception.
ns2={"torch":torch,"np":np,"F":torch.nn.functional,"Optional":Optional,"List":List}
extract("cdm/rewards/fid_scorer.py", {"FIDScorer"}, ns2)
scorer=ns2["FIDScorer"](device="cpu",batch_size=1)
captured=[]
def fake_inception(x):
    captured.append(x.clone())
    return torch.zeros((len(x),2048))
yy,xx=np.indices((1024,1024))
arr=np.repeat((((xx+yy)%2)*255).astype(np.uint8)[...,None],3,axis=2)
tensor=torch.tensor(arr.transpose(2,0,1).copy())[None].float()/255
scorer._extract_features_from_tensor_batch(fake_inception,tensor)
from cleanfid.resize import build_resizer
expected=build_resizer("clean")(arr)
got=captured[0][0].numpy().transpose(1,2,0)
checks["FID_preprocessing_checkerboard"]={"generated_vs_clean_PIL_mean_abs_difference_0_255":float(np.abs(got-expected).mean()),
                                       "generated_range":[float(got.min()),float(got.max())],
                                       "clean_PIL_range":[float(expected.min()),float(expected.max())]}
print(json.dumps(result,indent=2))
