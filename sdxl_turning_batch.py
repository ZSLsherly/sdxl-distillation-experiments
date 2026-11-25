"""Batched frozen-teacher trajectories; each prompt/seed remains an independent case."""
import argparse,gc,json,time
from pathlib import Path
import torch
from diffusers import UNet2DConditionModel,DDIMScheduler
from sdxl_turning_resample import observations

FIELDS=['rf_full','rf_conditional_on_guided_path','rf_spatial_centered','rf_coarse_centered']
METRICS=['angle_rad','rate_rad_per_u','path_rms_distance','rate_rad_per_path_rms','velocity_rms']

def metrics(old,new,du):
    values=[]
    for name in FIELDS:
        v,s=new[name];w,previous=old[name]
        v=v.flatten(1).double();w=w.flatten(1).double()
        nv=v.norm(dim=1);nw=w.norm(dim=1)
        angle=((v*w).sum(1)/(nv*nw).clamp_min(1e-30)).clamp(-1,1).acos()
        dist=(s.double()-previous.double()).flatten(1).square().mean(1).sqrt()
        valid=(nv>1e-10)&(nw>1e-10)&(dist>1e-10)
        values.append(torch.stack([angle,angle/du,dist,angle/dist.clamp_min(1e-30),v.square().mean(1).sqrt(),valid.double()],1))
    return torch.stack(values,1)

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--batch',type=int,default=8);p.add_argument('--rank',type=int,default=0);p.add_argument('--workers',type=int,default=1)
    p.add_argument('--probe',action='store_true');p.add_argument('--max-batches',type=int,default=0)
    p.add_argument('--memory-fraction',type=float,default=.58)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(a.memory_fraction)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    d=torch.load(a.root/'conditions512.pt',map_location='cpu',weights_only=True)
    assert len(d['prompts'])==128 and len(set(d['prompts']))==128
    base=a.root/'models/sdxl-base-1.0';sched=DDIMScheduler.from_pretrained(base,subfolder='scheduler',local_files_only=True)
    alpha=sched.alphas_cumprod.sqrt().cuda();sigma=(1-sched.alphas_cumprod).sqrt().cuda();u=(sigma/(alpha+sigma)).cpu().tolist()
    model=UNet2DConditionModel.from_pretrained(base,subfolder='unet',variant='fp16',torch_dtype=torch.bfloat16,local_files_only=True).eval().requires_grad_(False).cuda()
    cases=[(i,s) for i in range(128) for s in [201,202,203,204]]
    grid=sorted(set(list(range(999,-1,-10))+[0]),reverse=True)
    def inputs(group):
        ids=[i for i,s in group];b=len(ids)
        def c(key):
            z=d[key];return (z.expand(b,*z.shape[1:]) if len(z)==1 else z[ids]).to('cuda',dtype=torch.bfloat16)
        pe=torch.cat([c('negative_prompt_embeds'),c('prompt_embeds')])
        pp=torch.cat([c('negative_pooled_prompt_embeds'),c('pooled_prompt_embeds')])
        tids=c('time_ids').repeat(2,1)
        x=torch.cat([torch.randn((1,4,128,128),device='cuda',generator=torch.Generator(device='cuda').manual_seed(seed)) for i,seed in group])
        return x,pe,pp,tids
    def predict(x,t,pe,pp,tids):
        values=model(torch.cat([x,x]).to(torch.bfloat16),torch.tensor([t],device='cuda'),encoder_hidden_states=pe,
            added_cond_kwargs={'text_embeds':pp,'time_ids':tids},return_dict=False)[0].float()
        uncond,cond=values.chunk(2);return uncond+8*(cond-uncond),cond
    if a.probe:
        result=[];reference=None
        for b in [1,4,8,16]:
            try:
                with torch.inference_mode():
                    x,pe,pp,tids=inputs(cases[:b]);predict(x,999,pe,pp,tids);torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats();start=time.monotonic()
                    for t in [999,989,979]:
                        eps,_=predict(x,t,pe,pp,tids)
                        if t==999:first=eps[0].clone()
                        clean=(x-sigma[t]*eps)/alpha[t];x=alpha[t-10]*clean+sigma[t-10]*eps
                    torch.cuda.synchronize();elapsed=time.monotonic()-start
                    if reference is None:reference=first.clone()
                    error=((first-reference).square().mean().sqrt()/reference.square().mean().sqrt()).item()
                    row={'batch':b,'seconds_per_step':elapsed/3,'images_per_second_per_step':3*b/elapsed,
                        'peak_allocated_GiB':torch.cuda.max_memory_allocated()/2**30,'peak_reserved_GiB':torch.cuda.max_memory_reserved()/2**30,
                        'relative_rms_first_eps_vs_batch1':error}
                    result.append(row);print('PROBE',json.dumps(row),flush=True)
                    del x,pe,pp,tids,eps,clean,first
            except torch.OutOfMemoryError:
                print('PROBE_OOM',b,flush=True);break
            finally:gc.collect();torch.cuda.empty_cache()
        (a.out/'probe.json').write_text(json.dumps(result,indent=2));return
    selected=[c for index,c in enumerate(cases) if index%a.workers==a.rank]
    selected=[(i,s) for i,s in selected if not (a.out/f'base_dt10_bf16_prompt{i:03d}_seed{s}.json').exists()]
    completed=0
    with torch.inference_mode():
        for offset in range(0,len(selected),a.batch):
            group=selected[offset:offset+a.batch];x,pe,pp,tids=inputs(group);old=None;rows=[];tensors=[]
            torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.monotonic()
            for t,next_t in zip(grid,grid[1:]+[-1]):
                while (a.out/f'PAUSE_{a.rank}').exists():time.sleep(1)
                eps,cond=predict(x,t,pe,pp,tids)
                clean=(x-sigma[t]*eps)/alpha[t];clean_c=(x-sigma[t]*cond)/alpha[t]
                current=observations(eps-clean,cond-clean_c,x/(alpha[t]+sigma[t]))
                if old is not None:
                    pt,previous=old;du=abs(u[t]-u[pt]);tensors.append(metrics(previous,current,du))
                    rows.append({'t_from':pt,'t_to':t,'t_mid':(pt+t)/2,'u_mid':(u[t]+u[pt])/2,'delta_u':du})
                old=(t,current);x=clean if next_t<0 else alpha[next_t]*clean+sigma[next_t]*eps
            all_values=torch.stack(tensors).cpu().tolist();torch.cuda.synchronize();elapsed=time.monotonic()-start
            assert torch.isfinite(x).all().item(),'Nonfinite final latent'
            for j,(i,seed) in enumerate(group):
                sample_rows=[]
                for row,vals in zip(rows,all_values):
                    fields={k:{**dict(zip(METRICS,vals[j][f][:5])),'valid':bool(vals[j][f][5])} for f,k in enumerate(FIELDS)}
                    sample_rows.append({**row,'fields':fields})
                result={'model':'SDXL-base-1.0','mode':'base_dt10_bf16','prompt_index':i,'source_index':d['source_indices'][i],
                    'prompt':d['prompts'][i],'seed':seed,'cfg':8,'dtype':'torch.bfloat16','grid':grid,'rows':sample_rows,
                    'batch_size':len(group),'batch_seconds':elapsed,'seconds':elapsed/len(group),
                    'peak_allocated_GiB':torch.cuda.max_memory_allocated()/2**30,'worker':a.rank,
                    'coordinate':'u=sigma/(alpha+sigma),y=x/(alpha+sigma),v=eps-x0; path distance uses latent RMS',
                    'precision_note':'TF32 disabled; batch outputs can differ slightly from batch1 due to BF16 kernels'}
                dest=a.out/f'base_dt10_bf16_prompt{i:03d}_seed{seed}.json'
                dest.with_suffix('.tmp').write_text(json.dumps(result,allow_nan=False));dest.with_suffix('.tmp').replace(dest)
                if i<4:torch.save(x[j:j+1].cpu(),a.out/f'example_prompt{i:03d}_seed{seed}.pt')
            completed+=len(group);print('BATCH_DONE',a.rank,completed,len(group),round(elapsed,2),round(result['peak_allocated_GiB'],2),flush=True)
            del old,current,tensors,x,eps,cond,clean,clean_c
            if a.max_batches and completed>=a.max_batches*a.batch:break
    (a.out/f'worker{a.rank}_complete.json').write_text(json.dumps({'complete':True,'completed_this_run':completed,'rank':a.rank}))

if __name__=='__main__':main()
