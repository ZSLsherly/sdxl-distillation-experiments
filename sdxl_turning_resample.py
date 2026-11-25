"""Independent SDXL profiles in physical-noise and path-distance coordinates."""
import argparse,json,time,gc
from pathlib import Path
import torch
import torch.nn.functional as F
from diffusers import UNet2DConditionModel,DDIMScheduler,AutoencoderKL
from PIL import Image


def observations(velocity, conditional, state):
    centered=velocity-velocity.mean((2,3),keepdim=True)
    yc=state-state.mean((2,3),keepdim=True)
    return {'rf_full':(velocity,state),'rf_conditional_on_guided_path':(conditional,state),
            'rf_spatial_centered':(centered,yc),
            'rf_coarse_centered':(F.avg_pool2d(centered,8),F.avg_pool2d(yc,8))}


def turn_metrics(previous,current,delta_noise):
    result={}
    for key,(velocity,state) in current.items():
        old_velocity,old_state=previous[key]
        v,w=velocity.flatten().double(),old_velocity.flatten().double()
        nv,nw=v.norm(),w.norm()
        if min(nv.item(),nw.item())<1e-10:
            result[key]={'valid':False};continue
        angle=torch.acos((v.dot(w)/(nv*nw)).clamp(-1,1)).item()
        distance=(state.double()-old_state.double()).square().mean().sqrt().item()
        result[key]=dict(valid=True,angle_rad=angle,rate_rad_per_u=angle/delta_noise,
            path_rms_distance=distance,rate_rad_per_path_rms=angle/distance if distance>1e-10 else None,
            velocity_rms=velocity.double().square().mean().sqrt().item())
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--worker',type=int,default=0)
    ap.add_argument('--workers',type=int,default=1);ap.add_argument('--cfg',type=float,default=8.)
    ap.add_argument('--smoke',action='store_true',help='One BF16 trajectory and decoded example only')
    args=ap.parse_args();torch.set_num_threads(2);torch.cuda.set_device(0)
    if args.smoke:torch.cuda.set_per_process_memory_fraction(.60)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    args.out.mkdir(parents=True,exist_ok=True)
    data=torch.load(args.root/'conditions.pt',weights_only=True,map_location='cpu')
    base=args.root/'models/sdxl-base-1.0'
    scheduler=DDIMScheduler.from_pretrained(base,subfolder='scheduler',local_files_only=True)
    alpha=scheduler.alphas_cumprod.float().sqrt().cuda()
    sigma=(1-scheduler.alphas_cumprod.float()).sqrt().cuda();uu=sigma/(alpha+sigma)
    modes=[('base_dt10_bf16',10,32,[201,202],torch.bfloat16),
           ('fine_dt5_bf16',5,4,[201,202],torch.bfloat16),
           ('precision_dt10_fp32',10,2,[201,202],torch.float32)]
    if args.smoke:modes=[('base_dt10_bf16',10,1,[201],torch.bfloat16)]
    model=None;last_dtype=None
    for label,stride,count,seeds,dtype in modes:
        if dtype!=last_dtype:
            if model is not None:del model;gc.collect();torch.cuda.empty_cache()
            model=UNet2DConditionModel.from_pretrained(base,subfolder='unet',variant='fp16',
                torch_dtype=dtype,local_files_only=True).eval().requires_grad_(False).cuda()
            last_dtype=dtype
        grid=sorted(set(list(range(999,-1,-stride))+[0]),reverse=True)
        for case,(i,seed) in enumerate([(i,s) for i in range(count) for s in seeds]):
            if case%args.workers!=args.worker:continue
            dest=args.out/f'{label}_prompt{i:03d}_seed{seed}.json'
            if dest.exists():continue
            tick=time.monotonic()
            def cond(key):
                v=data[key];return (v[:1] if len(v)==1 else v[i:i+1]).to('cuda',dtype=dtype)
            pe=torch.cat([cond('negative_prompt_embeds'),cond('prompt_embeds')])
            pp=torch.cat([cond('negative_pooled_prompt_embeds'),cond('pooled_prompt_embeds')])
            tids=cond('time_ids').repeat(2,1)
            x=torch.randn((1,4,128,128),device='cuda',generator=torch.Generator(device='cuda').manual_seed(seed))
            rows=[];previous=None;torch.cuda.reset_peak_memory_stats()
            with torch.inference_mode():
                for t,next_t in zip(grid,grid[1:]+[-1]):
                    values=model(torch.cat([x,x]).to(dtype),torch.tensor([t],device='cuda'),
                        encoder_hidden_states=pe,added_cond_kwargs={'text_embeds':pp,'time_ids':tids},return_dict=False)[0].float()
                    uncond,conditional=values.chunk(2);eps=uncond+args.cfg*(conditional-uncond)
                    clean=(x-sigma[t]*eps)/alpha[t];clean_c=(x-sigma[t]*conditional)/alpha[t]
                    current=observations(eps-clean,conditional-clean_c,x/(alpha[t]+sigma[t]))
                    if previous is not None:
                        pt,old=previous;du=abs((uu[t]-uu[pt]).item())
                        rows.append(dict(t_from=pt,t_to=t,t_mid=(pt+t)/2,
                            u_mid=((uu[t]+uu[pt])/2).item(),delta_u=du,
                            fields=turn_metrics(old,current,du)))
                    previous=(t,{k:(v.clone(),s.clone()) for k,(v,s) in current.items()})
                    x=clean if next_t<0 else alpha[next_t]*clean+sigma[next_t]*eps
                if label=='base_dt10_bf16' and i<4:
                    torch.save({'prompt_index':i,'seed':seed,'latent':x.cpu()},
                        args.out/f'example_worker{args.worker}_prompt{i:03d}_seed{seed}.pt')
            result=dict(model='SDXL-base-1.0',mode=label,prompt_index=i,source_index=data['source_indices'][i],
                prompt=data['prompts'][i],seed=seed,cfg=args.cfg,dtype=str(dtype),grid=grid,
                seconds=time.monotonic()-tick,peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
                u_start=uu[999].item(),u_end=uu[0].item(),rows=rows,
                coordinate='u=sigma/(alpha+sigma),y=x/(alpha+sigma),v=eps-x0; C=identity primary',
                precision_note='TF32 disabled; both compute modes load same FP16 checkpoint')
            dest.with_suffix('.tmp').write_text(json.dumps(result));dest.with_suffix('.tmp').replace(dest)
            print('TRAJECTORY_DONE',label,i,seed,round(result['seconds'],2),round(result['peak_allocated_GiB'],2),flush=True)
    sampling_peak=torch.cuda.max_memory_allocated()/2**30
    del model;gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
    examples=sorted(args.out.glob(f'example_worker{args.worker}_*.pt'))
    if examples:
        vae=AutoencoderKL.from_pretrained(base,subfolder='vae',variant='fp16',torch_dtype=torch.float32,
            local_files_only=True).eval().requires_grad_(False).cuda()
        with torch.inference_mode():
            for file in examples:
                saved=torch.load(file,weights_only=True,map_location='cpu')
                i,seed,z=saved['prompt_index'],saved['seed'],saved['latent']
                if (args.out/f'example_prompt{i:03d}_seed{seed}.png').exists():continue
                rgb=vae.decode(z.cuda()/vae.config.scaling_factor,return_dict=False)[0]
                pixels=((rgb[0]/2+.5).clamp(0,1).permute(1,2,0)*255).round().byte().cpu().numpy()
                Image.fromarray(pixels).save(args.out/f'example_prompt{i:03d}_seed{seed}.png')
    (args.out/f'worker{args.worker}_complete.json').write_text(json.dumps({'complete':True,'worker':args.worker,
        'last_sampling_peak_allocated_GiB':sampling_peak,'decode_peak_allocated_GiB':torch.cuda.max_memory_allocated()/2**30}))
    print('WORKER_COMPLETE',args.worker,flush=True)


if __name__=='__main__':main()
