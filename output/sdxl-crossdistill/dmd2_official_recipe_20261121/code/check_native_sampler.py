from pathlib import Path
import json,os,sys,time,hashlib
import torch
from diffusers import UNet2DConditionModel,AutoencoderKL,DDIMScheduler
from PIL import Image
from run_guard import check_idle
from objectives import VPSchedule
from native_sample_reference import sample
r=Path(sys.argv[1]);cfg=json.loads((r/'config.json').read_text());uid=os.environ['CUDA_VISIBLE_DEVICES'];check_idle([uid])
torch.set_num_threads(2);device='cuda';run=r/'runs/formal_g160_restart_20261120';out=run/'native_sampler_check';out.mkdir(exist_ok=True)
weights=torch.load(run/'student.pt',map_location='cpu',weights_only=True,mmap=True)
with torch.device('meta'):net=UNet2DConditionModel.from_config(json.loads((Path(cfg['model_id'])/'unet/config.json').read_text()))
net.load_state_dict(weights,strict=True,assign=True);net=net.float().to(device).eval().requires_grad_(False);del weights
vae=AutoencoderKL.from_pretrained(cfg['model_id'],subfolder='vae',variant='fp16',torch_dtype=torch.float32,local_files_only=True).to(device).eval().requires_grad_(False)
scheduler=DDIMScheduler.from_pretrained(cfg['model_id'],subfolder='scheduler',local_files_only=True);schedule=VPSchedule(scheduler.alphas_cumprod.to(device))
data=torch.load(Path(cfg['data_dir'])/'prepared.pt',map_location='cpu',weights_only=True,mmap=True)
cases=[]
for p in sorted((run/'eval/g000160').glob('rank*.json')):cases+=json.loads(p.read_text())['samples']
class AMPModel(torch.nn.Module):
 def forward(self,*a,**kw):
  with torch.autocast('cuda',dtype=torch.bfloat16):res=net(*a,**kw)
  return res
model=AMPModel();records=[]
with torch.no_grad():
 for case in cases:
  i=data['original_indices'].index(case['original_index']);emb=data['prompt_embeds'][i:i+1].to(device,dtype=torch.float32);pool=data['pooled_prompt_embeds'][i:i+1].to(device,dtype=torch.float32)
  cond={'time_ids':data['time_ids'].to(device,dtype=torch.float32),'text_embeds':pool}
  for mode in ['native','ddim_replay']:
   torch.manual_seed(case['seed']);noise=torch.randn(1,4,128,128,device=device)
   if mode=='native':
    img=sample(noise,cond,model,vae,scheduler,emb,device=device,num_step=4)[0].cpu().numpy()
   else:
    x=noise
    for t,nt in zip([999,749,499,249],[749,499,249,-1]):
     tt=torch.tensor([t],device=device);eps=model(x,tt,emb,added_cond_kwargs=cond).sample.float();x=schedule.jump(x,eps,tt,nt)
    rgb=vae.decode(x/vae.config.scaling_factor).sample;assert torch.isfinite(rgb).all()
    img=((rgb[0]/2+.5).clamp(0,1).permute(1,2,0)*255).round().byte().cpu().numpy()
   p=out/f'{mode}_{case["image"]}';Image.fromarray(img).save(p)
   records.append(dict(case,mode=mode,file=p.name));print('SAVED',p.name,flush=True)
(out/'complete.json').write_text(json.dumps({'count':len(records),'records':records,'source':'official main/sdxl/test_folder_sdxl.py sample() and get_x0_from_noise() extracted unchanged; BF16 UNet forward, FP32 VAE','checkpoint':str(run/'student.pt'),'time':time.time()},indent=2))
