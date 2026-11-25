source /data5/shenglin.zeng/env.sh
python - <<'PY'
import sys,torch
sys.path.insert(0,'/data5/shenglin.zeng/sdxl_crossdistill/src')
from diffusers import DDIMScheduler
from sdxl_objectives import VPSchedule
schedule=VPSchedule(DDIMScheduler.from_pretrained('/data5/shenglin.zeng/sdxl_crossdistill/models/sdxl-base-1.0',subfolder='scheduler',local_files_only=True).alphas_cumprod)
x=torch.randn(1,4,8,8);eps=torch.randn_like(x)
for t,s in [(999,979),(949,899),(899,749),(749,599),(599,0)]:
 a,b=schedule.coefficients(t,x);aa,bb=schedule.coefficients(s,x)
 y=x/(a+b);u=b/(a+b);uu=bb/(aa+bb);v=eps-schedule.clean(x,eps,t)
 actual=schedule.jump(x,eps,t,s)/(aa+bb)
 assert torch.allclose(actual,y+(uu-u)*v,atol=3e-6)
print('PASS: native VP/DDIM jump equals RF-coordinate Euler velocity epsilon-x0; SNR preserved')
PY
