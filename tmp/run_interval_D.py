import json,fcntl,os,shutil,subprocess,sys,time
from pathlib import Path
from run_quota_bc import verify
from launch import command,gpu_check
from turning_batch64_workflow import descendants,foreign_on,stop_own
from turning_sampling import TurningSampler
root=Path(os.environ['SDXL_ROOT'])
code=Path(__file__).resolve().parent
out=root/'outputs/interval_D_20261103'; out.mkdir(exist_ok=True)
lock=(out/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
cfg=json.loads((code/'configs/turning_D_quota_a800_g160.json').read_text())
c=json.loads((code/'configs/turning_C_quota_a800_g160.json').read_text())
assert dict(cfg,turning_profile=c['turning_profile'])==c
D=TurningSampler(root/cfg['turning_profile'],979)
C=TurningSampler(root/c['turning_profile'],979)
for lo,hi in [(509,979),(259,979),(20,979)]:
    t,p=D.distribution((lo,hi))
    masses=[sum(v for i,v in zip(t,p) if i>=900),sum(v for i,v in zip(t,p) if i<=100),sum(v for i,v in zip(t,p) if 100<i<900)]
    want=(.3,0,.7) if lo>20 else (.2,.1,.7)
    assert all(abs(x-y)<1e-12 for x,y in zip(masses,want))
    if lo>20: assert (t,p)==C.distribution((lo,hi))
    print('PROBABILITIES',lo,hi,masses,flush=True)
assert not (out/'D').exists()
assert shutil.disk_usage(out).free>8*1024**3
cards=gpu_check([0],76)
env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TORCHDYNAMO_DISABLE='1')
with (out/'D.log').open('x') as log:
    p=subprocess.Popen(command(cfg,root,out.name+'/D',1),cwd=code,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    print('STARTED D',p.pid,cards,flush=True)
    try:
        while p.poll() is None:
            if foreign_on({cards[0][1]},descendants(p.pid)) or (out/'USER_STOP').exists():raise RuntimeError('GPU conflict or stop requested')
            time.sleep(10)
        if p.returncode:raise RuntimeError('D failed; see D.log')
    except BaseException:stop_own(p);raise
w=Path(verify(out/'D'))/'student_rank0.pt'
import torch
from diffusers import UNet2DConditionModel
state=torch.load(w,map_location='cpu',mmap=True,weights_only=True)
with torch.device('meta'):m=UNet2DConditionModel.from_config(json.loads((root/'models/sdxl-base-1.0/unet/config.json').read_text()))
m.load_state_dict(state,strict=True,assign=True)
assert all(torch.isfinite(v).all() for v in state.values())
(out/'complete.json').write_text(json.dumps(dict(G=160,fake=320,weight=str(w),finite=True,strict_load=True)))
