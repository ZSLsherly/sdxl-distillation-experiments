import json
from pathlib import Path
import torch
from stage_intervention_core import interval_grid,integrate,student_rollout,replace_stage
from sdxl_parallel.objectives import VPSchedule
schedule=VPSchedule(torch.linspace(.999,.01,1000));knots=[999,749,499,249,-1]
torch.manual_seed(123);clean=torch.randn(2,4,8,8);noise=torch.randn_like(clean)
start=schedule.noise(clean,noise,999)
oracle=lambda x,t:noise
states=student_rollout(start,knots,oracle,schedule.jump)
errors=[]
for k in range(4):
 final,boundary=replace_stage(states,k,knots,oracle,oracle,schedule.jump)
 errors.append(float((final-clean).abs().max()))
 expected=schedule.noise(clean,noise,knots[k+1])
 assert torch.allclose(boundary,expected,atol=1e-5)
 assert torch.allclose(final,clean,atol=1e-5)
 assert torch.equal(states[0],start)
assert interval_grid(249,-1)[-2:]==[0,-1]
assert interval_grid(999,749)[-1]==749
assert sum(len(interval_grid(knots[k],knots[k+1]))-1 for k in range(4))==101
out=Path('output/sdxl-crossdistill/stage_intervention_20261026');out.mkdir(parents=True,exist_ok=True)
(out/'cpu_check.json').write_text(json.dumps({'passed':True,'max_final_error_by_stage':errors,'teacher_total_calls':101},indent=2))
print((out/'cpu_check.json').read_text())
