import json,tempfile,subprocess
from pathlib import Path
from unittest.mock import Mock,patch
import torch
import run_dm1_uniform_control as q
import boundary as b
from objectives import VPSchedule
from diffusers import DDIMScheduler

q.verify_stage_losses(q.base/'model')
with tempfile.TemporaryDirectory() as name:
    out=Path(name)
    rows=[json.loads(l) for l in (q.base/'model/rank0.jsonl').read_text().splitlines()]
    rows=[r for r in rows if r['phase']=='student_done']
    rows[90]['first_dm_loss']=float('nan')
    (out/'rank0.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
    try:q.verify_stage_losses(out)
    except AssertionError:pass
    else:raise AssertionError('non-finite first DM accepted')
    for conflict in (False,True):
        p=Mock(pid=123,returncode=0); p.poll.side_effect=[None,None,0]
        with patch.object(q,'out',out),patch.object(q,'gpu_check',return_value=[[0,'gpu']]),patch.object(q.subprocess,'Popen',return_value=p),patch.object(q,'foreign_on',side_effect=[subprocess.TimeoutExpired('query',10),[999] if conflict else []]),patch.object(q,'descendants',return_value={123}),patch.object(q.time,'sleep'),patch.object(q,'stop_own') as stop:
            try:q.run_gpu(['dummy'],out/'worker.log')
            except RuntimeError:assert conflict
            assert stop.call_count==int(conflict)
knots=[999,749,499,249,-1]
sched=VPSchedule(DDIMScheduler(beta_start=.00085,beta_end=.012,beta_schedule='scaled_linear').alphas_cumprod)
torch.manual_seed(7)
for mid in (2,3):
    x=torch.randn(2,4,4,4)
    w=torch.nn.Parameter(torch.tensor([.01,.02,.03,.04]))
    def predict(x,t,c):return w[knots.index(t[0].item())]*x
    seen=[]
    for k,y in b.local_rollout(predict,sched,None,x,knots,(1,mid)):
        expected,_=b.rollout_boundary_and_final(predict,sched,None,x,knots,k)
        torch.testing.assert_close(y.detach(),expected)
        t=torch.tensor([965,970]);eps=torch.randn_like(y)
        xt=b.renoise(sched,y.detach(),knots[k],t,eps)
        loss,_=b.dm_surrogate(sched,y,knots[k],t,xt,.04*xt,.06*xt)
        grad=torch.autograd.grad(loss,w)[0]
        assert grad.nonzero().flatten().tolist()==[k-1]
        seen.append(k)
    assert seen==[1,mid,4]
print('CPU_OK: parent 160 student records; NaN rejected; query timeout versus real conflict; both middle branches and isolated local gradients')

