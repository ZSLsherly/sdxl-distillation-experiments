import json,sys,os,math
from pathlib import Path
import torch
import boundary as b
from train import arguments
from launch import flags
from turning_sampling import TurningSampler
root=Path(os.environ['SDXL_ROOT'])
out=root/'outputs/c_mix1_first_dm_fake3_g160'
cfg=json.loads((out/'effective_config.json').read_text())
sys.argv=['train','--run','validate',*flags(cfg)]
args=arguments()
assert args.fake_updates==3 and args.pcm_weight==0 and not args.both_middle and args.fake_student_mix==.01
assert args.batch==64 and args.micro_batch==16 and args.max_student_updates==160 and args.variant=='dmd_only'
knots=[*args.schedule,-1]
ranges=b.phase_ranges(knots,10,979)
assert ranges=={1:(759,979),2:(509,979),3:(259,979),4:(20,979)}
times,prob=TurningSampler(root/cfg['turning_profile'],979).distribution(ranges[1])
assert math.isclose(sum(p for t,p in zip(times,prob) if t>=900),.30)
for g in range(160):
    stages=[b.critic_stage(3*g+i,True,args.seed,3) for i in range(3)]
    assert stages==[1,b.middle_stage(g,args.seed),4]
class Schedule:
    def jump(self,x,p,t,target): return x+p
initial=torch.ones(2,1)
for mid in (2,3):
    weights=torch.nn.Parameter(torch.ones(4))
    def predict(x,t,c):return weights[knots.index(int(t[0]))]*x
    first,final=b.rollout_boundary_and_final(predict,Schedule(),None,initial,knots,1)
    assert torch.equal(first,initial*2) and torch.equal(final,initial*16)
    gradients=[]
    for k,y in b.local_rollout(predict,Schedule(),None,initial,knots,(1,mid)):
        gradients.append(k)
        grad,=torch.autograd.grad(y.sum(),weights)
        assert grad.nonzero().flatten().tolist()==[k-1]
    assert gradients==[1,mid,4]
assert [int(b.stage_labels(i,initial)[0]) for i in (1,2,3,4)]==[3,0,1,2]
print('CPU validated: 160 paired schedules; first x749; local gradients; first prior 30%; parsed launch config')

