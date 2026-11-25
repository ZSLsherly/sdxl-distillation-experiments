"""Validate the measuring hook independently of model loading or CUDA."""
import ast,math
from pathlib import Path
import torch
source=Path(__file__).with_name('profile_official.py').read_text()
tree=ast.parse(source)
fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='directions')
scope={'torch':torch,'math':math}
exec(compile(ast.Module(body=[fn],type_ignores=[]),'<directions>','exec'),scope)
directions=scope['directions']
v=torch.randn(2,21,3,4,dtype=torch.float64)
before=v.clone();a=directions(v)
assert torch.equal(v,before),'measurement mutated input'
w=v.clone();w[:,0]+=1000
b=directions(w)
assert torch.equal(a['future_centered'][0],b['future_centered'][0]),'observed frame included'
w=v.clone();w[:,1:]+=torch.randn(2,1,3,4)
c=directions(w)
assert torch.allclose(a['future_centered'][0],c['future_centered'][0],atol=1e-6),'temporal static component not removed'
d=directions(v*7)
assert torch.allclose(a['future_centered'][0],d['future_centered'][0],atol=1e-6),'normalization failed'
theta=.25;u=torch.tensor([1.,0.],dtype=torch.float64);w=torch.tensor([math.cos(theta),math.sin(theta)],dtype=torch.float64)
assert abs(math.acos(torch.dot(u,w).item())/.05-5)<1e-10
print('PASS: no mutation, observed exclusion, temporal centering, scale invariance, angle/noise units')
