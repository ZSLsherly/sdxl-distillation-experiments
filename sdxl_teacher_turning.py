"""DyMD-inspired image teacher turning: RF-equivalent velocity, C=identity.

SDXL VP: x=a*x0+s*eps. Rescale y=x/(a+s), u=s/(a+s), so
y=(1-u)*x0+u*eps and v_RF=eps-x0. Single-image temporal centering
would annihilate the vector; spatial centering is explicitly an ablation.
"""
import json,os,gc,time
from pathlib import Path
import torch
import torch.nn.functional as F
from diffusers import UNet2DConditionModel,DDIMScheduler
from sdxl_objectives import VPSchedule

@torch.inference_mode()
def main():
    rank=int(os.environ.get('LOCAL_RANK',0));torch.cuda.set_device(rank);torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=True
    root=Path('/data5/shenglin.zeng/sdxl_crossdistill');out=root/'outputs/hparam_pilot/teacher_turning';out.mkdir(parents=True,exist_ok=True)
    base=root/'models/sdxl-base-1.0';data=torch.load(root/'datasets/prepared.pt',weights_only=True,map_location='cpu')
    schedule=VPSchedule(DDIMScheduler.from_pretrained(base,subfolder='scheduler',local_files_only=True).alphas_cumprod)
    alpha=schedule.alpha;sigma=schedule.sigma;u=sigma/(alpha+sigma)
    model=UNet2DConditionModel.from_pretrained(base,subfolder='unet',variant='fp16',torch_dtype=torch.bfloat16,local_files_only=True).eval().requires_grad_(False).to('cuda')
    modes=[('base_dt10_bf16',10,16,torch.bfloat16),('fine_dt5_bf16',5,8,torch.bfloat16),('precision_dt10_fp32',10,4,torch.float32)]
    for label,stride,prompts,dtype in modes:
        if dtype==torch.float32:
            del model;gc.collect();torch.cuda.empty_cache()
            model=UNet2DConditionModel.from_pretrained(base,subfolder='unet',variant='fp16',torch_dtype=torch.float32,local_files_only=True).eval().requires_grad_(False).to('cuda')
        model.to(dtype=dtype);records=[];grid=list(range(999,-1,-stride))+[0]
        grid=sorted(set(grid),reverse=True);start=time.perf_counter()
        for idx in range(96+rank,96+prompts,4):
            pe=data['prompt_embeds'][idx:idx+1].to('cuda',dtype=dtype);ne=data['negative_prompt_embeds'][idx:idx+1].to('cuda',dtype=dtype)
            pp=data['pooled_prompt_embeds'][idx:idx+1].to('cuda',dtype=dtype);np=data['negative_pooled_prompt_embeds'][idx:idx+1].to('cuda',dtype=dtype)
            tids=data['time_ids'][idx:idx+1].to('cuda',dtype=dtype).repeat(2,1)
            for seed in [101,102]:
                x=torch.randn((1,4,128,128),device='cuda',generator=torch.Generator(device='cuda').manual_seed(seed))
                previous=None;rows=[];snapshots={};norms=[]
                for t,s in zip(grid,grid[1:]+[-1]):
                    tt=torch.tensor([t],device='cuda',dtype=torch.long)
                    values=model(torch.cat([x,x]).to(dtype),tt,encoder_hidden_states=torch.cat([ne,pe]),added_cond_kwargs={'text_embeds':torch.cat([np,pp]),'time_ids':tids},return_dict=False)[0].float()
                    eu,ec=values.chunk(2);eg=eu+8*(ec-eu)
                    clean=schedule.clean(x,eg,tt);clean_c=schedule.clean(x,ec,tt)
                    velocity=eg-clean;centered=velocity-velocity.mean((2,3),keepdim=True)
                    fields={'rf_full':velocity,'rf_spatial_centered':centered,'rf_coarse_centered':F.avg_pool2d(centered,8),
                            've_epsilon':eg,'rf_conditional_on_guided_path':ec-clean_c}
                    directions={};magnitudes={}
                    for key,value in fields.items():
                        flat=value.flatten().double();norm=flat.norm();assert norm>1e-9
                        directions[key]=flat/norm;magnitudes[key]=norm.item()
                    norms.append({'t':t,'u':u[t].item(),'norms':magnitudes})
                    if previous is not None:
                        pt,units=previous;du=abs(u[t].item()-u[pt].item());angles={}
                        for key,unit in directions.items():
                            angle=torch.acos((unit*units[key]).sum().clamp(-1,1)).item()
                            angles[key]={'angle_rad':angle,'rate_rad_per_u':angle/du}
                        rows.append({'t_from':pt,'t_to':t,'t_mid':(pt+t)/2,'u_mid':(u[pt].item()+u[t].item())/2,'delta_u':du,'fields':angles})
                    previous=(t,directions)
                    if label=='base_dt10_bf16' and idx==96+rank and seed==101:snapshots[t]=clean.cpu().half()
                    x=schedule.jump(x,eg,tt,s)
                if snapshots:torch.save({'clean_estimates':snapshots,'final':x.cpu(),'cache_index':idx,'seed':seed,'prompt':data['prompts'][idx]},out/f'example_rank{rank}.pt')
                records.append({'cache_index':idx,'seed':seed,'prompt':data['prompts'][idx],'rows':rows,'norms':norms})
                print('TURNING_TRAJECTORY_DONE',label,idx,seed,flush=True)
        (out/f'{label}_rank{rank}.json').write_text(json.dumps({'label':label,'dtype':str(dtype),'stride':stride,'records':records,'seconds':time.perf_counter()-start,'teacher_cfg':8,'grid':grid,
            'rf_coordinate':'u=sigma_VP/(alpha_VP+sigma_VP); y=x_VP/(alpha_VP+sigma_VP); v_RF=epsilon_hat-x0_hat',
            'centering':'C=identity for primary rf_full; subtract per-channel spatial mean only in explicitly named image ablations',
            'source':'https://arxiv.org/html/2609.31349v1#S3.SS2','not_dymd_reproduction':'This profiles an SDXL image teacher; it does not implement video temporal centering, affinity adaptation or critic reweighting.'}))
        print('TURNING_MODE_COMPLETE',label,rank,flush=True)
    print('TEACHER_TURNING_COMPLETE',rank,flush=True)

if __name__=='__main__':main()
