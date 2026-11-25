"""Two-GPU, accumulated-batch fresh SDXL experiment with timed weight snapshots."""
import argparse,gc,hashlib,json,math,os,random,shutil,time
from pathlib import Path
from datetime import datetime,timedelta,timezone
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed._composable.fsdp import fully_shard,MixedPrecisionPolicy,CPUOffloadPolicy
from diffusers import UNet2DConditionModel,AutoencoderKL,DDIMScheduler
from diffusers.models.attention import BasicTransformerBlock
from diffusers.models.resnet import ResnetBlock2D,Downsample2D,Upsample2D
from diffusers.models.transformers.transformer_2d import Transformer2DModel
from sdxl_objectives import VPSchedule,dmd_surrogate,verify_math
from sdxl_pixel_gan import decode_tile,discriminator_loss,average_gradients
from sdxl_pixel_v4 import PixelDinoDiscriminator

def local(x):return x.to_local() if hasattr(x,'to_local') else x
def stamp():return datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%dT%H%M%S%z')
def atomic(path,obj):
    p=path.with_suffix(path.suffix+'.tmp');p.write_text(json.dumps(obj,indent=2,ensure_ascii=False));p.replace(path)

def decision(value,device):
    flag=torch.tensor(int(bool(value)) if dist.get_rank()==0 else 0,device=device)
    dist.broadcast(flag,src=0);return bool(flag.item())

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--variant',choices=['gan4','joint4'],required=True)
    ap.add_argument('--batch',type=int,default=64);ap.add_argument('--hours',type=float,default=24)
    ap.add_argument('--preflight',action='store_true');ap.add_argument('--offload-student',action='store_true')
    ap.add_argument('--max-student-updates',type=int)
    args=ap.parse_args();rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);gpu=int(os.environ['LOCAL_RANK'])
    assert world==2 and args.batch%world==0
    torch.cuda.set_device(gpu);dev=torch.device('cuda',gpu);torch.set_num_threads(4)
    dist.init_process_group('nccl',timeout=timedelta(minutes=30))
    root=Path('/data5/shenglin.zeng/sdxl_crossdistill');base=root/'models/sdxl-base-1.0';out=root/'outputs'/args.run;out.mkdir(parents=True,exist_ok=True)
    assert not (out/f'rank{rank}.jsonl').exists(),'Refusing to overwrite a run'
    (out/f'pid_rank{rank}').write_text(str(os.getpid()))
    accum=2 if args.preflight else args.batch//world;effective=world*accum
    data=torch.load(root/'datasets/gan_v3/prepared.pt',map_location='cpu',weights_only=True,mmap=True)
    manifest=json.loads((root/'datasets/gan_v3/manifest.json').read_text());n=data['train_count']
    order=random.Random(20261017).sample(range(n),n)
    mesh=init_device_mesh('cuda',(world,));policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16,reduce_dtype=torch.float32,cast_forward_inputs=True)
    verify_math();torch.manual_seed(20261017)
    def load(train,offload=False):
        m=UNet2DConditionModel.from_pretrained(base/'unet',variant='fp16',torch_dtype=torch.float32 if train else torch.bfloat16,local_files_only=True).to(dev)
        m.requires_grad_(train);m.train(train)
        if train:
            m.enable_gradient_checkpointing();extra={'offload_policy':CPUOffloadPolicy()} if offload else {}
            for child in reversed(list(m.modules())):
                if isinstance(child,(BasicTransformerBlock,ResnetBlock2D,Transformer2DModel,Downsample2D,Upsample2D)):fully_shard(child,mesh=mesh,mp_policy=policy,**extra)
            fully_shard(m,mesh=mesh,mp_policy=policy,**extra)
        return m
    print('LOAD',rank,flush=True)
    student=load(True,args.offload_student);fake=load(True,True);teacher=load(False)
    vae=AutoencoderKL.from_pretrained(base,subfolder='vae',variant='fp16',torch_dtype=torch.float32,local_files_only=True).to(dev).eval().requires_grad_(False)
    torch.manual_seed(20261017+7000);disc=PixelDinoDiscriminator(root/'models/dinov2-small').to(dev);disc.set_heads(False)
    og=torch.optim.AdamW(student.parameters(),lr=1e-6,betas=(.9,.999),weight_decay=.01,foreach=False)
    of=torch.optim.AdamW(fake.parameters(),lr=1e-6,betas=(.9,.999),weight_decay=.01,foreach=False)
    od=torch.optim.AdamW(disc.trainable(),lr=1e-4,betas=(0,.99),weight_decay=0,foreach=False)
    sched=VPSchedule(DDIMScheduler.from_pretrained(base,subfolder='scheduler',local_files_only=True).alphas_cumprod)
    knots=[999,919,499,199,-1]
    def cond(idx):
        return {k:(data[k][idx:idx+1] if not k.startswith('negative') and k!='time_ids' else data[k][:1]).to(dev) for k in ['prompt_embeds','negative_prompt_embeds','pooled_prompt_embeds','negative_pooled_prompt_embeds','time_ids']}
    def pred(m,x,t,c,neg=False):
        with torch.autocast('cuda',dtype=torch.bfloat16):return m(x,t,encoder_hidden_states=c['negative_prompt_embeds' if neg else 'prompt_embeds'],added_cond_kwargs={'text_embeds':c['negative_pooled_prompt_embeds' if neg else 'pooled_prompt_embeds'],'time_ids':c['time_ids']},return_dict=False)[0].float()
    def guide(x,t,c):
        u=pred(teacher,x,t,c,True);return u+8*(pred(teacher,x,t,c)-u)
    def rollout(c,stage=4,grad=False,noise=None):
        x=torch.randn((1,4,128,128),device=dev) if noise is None else noise.clone()
        for i in range(stage):
            with torch.enable_grad() if grad and i==stage-1 else torch.no_grad():
                t=torch.tensor([knots[i]],device=dev);x=sched.jump(x,pred(student,x,t,c),t,knots[i+1] if i<stage-1 else -1)
            if i<stage-1:x=x.detach()
        return x
    def normclip(m):
        total=torch.zeros((),device=dev,dtype=torch.float64);count=0
        for p in m.parameters():
            if p.grad is not None:
                v=local(p.grad);assert torch.isfinite(v).all();total+=v.double().square().sum().to(dev);count+=1
        dist.all_reduce(total);assert count
        value=total.sqrt().item()
        for p in m.parameters():
            if p.grad is not None:p.grad.mul_(min(1,10/(value+1e-12)))
        return value
    config={**vars(args),'effective_batch':effective,'microbatch_per_gpu':1,'accumulation':accum,'teacher_cfg':8,'student_cfg':1,'schedule':knots,'train_count':n,'fake_updates_per_student':5,'student_lr':1e-6,'fake_lr':1e-6,'D_lr':1e-4,'D':'frozen DINOv2-S/14 layers3,6,9,12 with 4 trainable heads + RGB conditional PatchGAN, equal logits blend','D_real':'native VAE reconstructed LAION tiles, 256px; DINO center224','initialization':'untouched SDXL Base for student/fake/teacher; fresh common D heads','cpu_offload':'fake parameters/gradients/optimizer; student additionally if configured','eval_every_student_samples':128,'snapshot_seconds':1800,'timezone':'Asia/Shanghai','stage_policy':'round-robin2,3,4; high-noise PCM each student update; shared UNet parameters','deadline_hours':args.hours}
    if rank==0:
        atomic(out/'config.json',config);atomic(out/'data_manifest.json',manifest)
        for name in ['sdxl_gan_train_v4.py','sdxl_pixel_v4.py','sdxl_pixel_gan.py','sdxl_objectives.py']:shutil.copyfile(Path(__file__).with_name(name),out/name)
        atomic(out/'source_sha256.json',{name:hashlib.sha256((out/name).read_bytes()).hexdigest() for name in ['sdxl_gan_train_v4.py','sdxl_pixel_v4.py','sdxl_pixel_gan.py','sdxl_objectives.py']})
    log=(out/f'rank{rank}.jsonl').open('w');gcount=fcount=dcount=0;start=time.monotonic();last_weight=start
    calibration=root/'outputs/gan_v3_calibration.json'
    weight=json.loads(calibration.read_text())['gan_weight'] if calibration.exists() else .003530974732711911
    if not args.preflight:assert calibration.exists(),'Require successful v3 gradient calibration'
    if rank==0:
        config.update(revision='4-lean',checkpoint_kind='student BF16 weights only; no fake/D/optimizer states and no exact-resume support',GAN_weight=weight)
        atomic(out/'config.json',config)
    torch.cuda.reset_peak_memory_stats()
    def snapshot(reason,phase,micro):
        nonlocal last_weight
        names=[stamp() if rank==0 else None];dist.broadcast_object_list(names,src=0);dest=out/'weights'/names[0];dest.mkdir(parents=True,exist_ok=True)
        state={k:local(v).detach().to(torch.bfloat16).cpu() for k,v in student.state_dict().items()}
        torch.save(state,dest/f'student_rank{rank}.pt');del state
        atomic(dest/f'shapes_rank{rank}.json',{k:{'shape':list(v.shape),'placements':[str(p) for p in v.placements] if hasattr(v,'placements') else []} for k,v in student.state_dict().items()})
        dist.barrier()
        if rank==0:atomic(dest/'complete.json',{'timestamp':names[0],'reason':reason,'student_updates':gcount,'fake_updates':fcount,'D_updates':dcount,'phase':phase,'micro_finished':micro,'weight_kind':'student BF16 inference shards only; not an optimizer-resume checkpoint','world_size':world,'gan_weight':weight})
        last_weight=time.monotonic();print('WEIGHTS_SAVED',rank,str(dest),gcount,flush=True)
    def stop_due():
        return (out.parent/'STOP_REQUEST').exists() or (out/'STOP_REQUEST').exists() or time.monotonic()-start>args.hours*3600
    def boundary_actions():
        return decision(stop_due() if rank==0 else False,dev)
    @torch.no_grad()
    def evaluate(tag):
        rng=torch.get_rng_state();crng=torch.cuda.get_rng_state();names=[stamp() if rank==0 else None];dist.broadcast_object_list(names,src=0);dest=out/'eval'/f'{names[0]}_g{gcount:06d}_{tag}';dest.mkdir(parents=True,exist_ok=True)
        # Both ranks execute exactly the same number of FSDP forwards.
        hold=[data['original_indices'].index(i) for i in manifest['eval_original_indices']]
        cases=[('train',order[rank]),('train',order[rank+2]),('holdout',hold[rank]),('holdout',hold[rank+2])]
        metrics=[]
        for split,idx in cases:
            c=cond(idx);noise=torch.randn((1,4,128,128),device=dev,generator=torch.Generator(device=dev).manual_seed(102));x=rollout(c,noise=noise)
            assert torch.isfinite(x).all()
            full_fake=vae.decode(x.float()/vae.config.scaling_factor,return_dict=False)[0].cpu()
            rgb=(full_fake/2+.5).clamp(0,1)[0]
            from PIL import Image
            import numpy as np
            arr=(rgb.permute(1,2,0).numpy()*255).round().astype(np.uint8);name=f'{split}_original{data["original_indices"][idx]}_seed102.png';Image.fromarray(arr).save(dest/name)
            gray=arr.astype(np.float32).mean(-1)/255;lap=-4*gray[1:-1,1:-1]+gray[:-2,1:-1]+gray[2:,1:-1]+gray[1:-1,:-2]+gray[1:-1,2:]
            item={'split':split,'original_index':data['original_indices'][idx],'prompt':data['prompts'][idx],'seed':102,'image':name,'laplacian_variance_diagnostic_only':float(lap.var()),'latent_std':x.std().item(),'D':{},'decode_comparison':{}}
            real=data['latents'][idx:idx+1].to(dev,dtype=torch.float32)
            for kind,latent in [('fake',x),('real',real)]:
                tile=decode_tile(vae,latent,(40,40));parts=disc(tile,c['pooled_prompt_embeds'],return_parts=True)
                item['D'][kind]={k:parts[k].mean().item() for k in ['total','dino','rgb']}
                item['D'][kind]['dino_heads']=parts['dino_heads'][:,0].tolist()
                if gcount==0:
                    full=full_fake if kind=='fake' else vae.decode(latent/vae.config.scaling_factor,return_dict=False)[0].cpu()
                    crop=full[:,:,384:640,384:640].to(dev)
                    item['decode_comparison'][kind]={'rgb01_mae':((tile-crop).abs().mean()/2).item(),'rgb01_rmse':((tile-crop).square().mean().sqrt()/2).item()}
                    panels=torch.cat([(crop/2+.5).clamp(0,1),(tile/2+.5).clamp(0,1),((tile-crop).abs()/2*4).clamp(0,1)],dim=3)[0].cpu()
                    png=(panels.permute(1,2,0).numpy()*255).round().astype(np.uint8)
                    Image.fromarray(png).save(dest/f'{split}_original{data["original_indices"][idx]}_{kind}_full_tile_absdiffx4.png')
                    del crop,full
                del tile,parts
            metrics.append(item);del full_fake,real,x
        atomic(dest/f'rank{rank}.json',{'student_updates':gcount,'student_samples':gcount*effective,'samples':metrics})
        torch.set_rng_state(rng);torch.cuda.set_rng_state(crng);dist.barrier();print('EVALUATED',rank,gcount,flush=True)
    def timed(phase,micro):
        if decision(time.monotonic()-last_weight>=1800 if rank==0 else False,dev):snapshot('half_hour',phase,micro)
        if rank==0:atomic(out/'status.json',{'status':'running','timestamp':stamp(),'student_updates':gcount,'student_samples':gcount*effective,'fake_updates':fcount,'D_updates':dcount,'phase':phase,'micro_finished':micro,'micro_total':accum,'effective_batch':effective,'elapsed_seconds':time.monotonic()-start,'peak_allocated_GiB':torch.cuda.max_memory_allocated()/1024**3})
    evaluate('initial')
    target=args.max_student_updates if args.max_student_updates is not None else (2 if args.preflight else 10**9)
    while gcount<target:
        if decision(stop_due() if rank==0 else False,dev):break
        stage=4 if args.preflight else 2+gcount%3
        variant=('joint4' if gcount==0 else 'gan4') if args.preflight else args.variant
        critic_repeats=1 if args.preflight else 5
        stopping=False
        for fi in range(critic_repeats):
            of.zero_grad(set_to_none=True);od.zero_grad(set_to_none=True);disc.set_heads(True)
            losses=[];d_records=[]
            for micro in range(accum):
                sequence=fcount*effective+micro*world+rank;idx=order[sequence%n];c=cond(idx);real=data['latents'][idx:idx+1].to(dev,dtype=torch.float32)
                torch.manual_seed(20261017+sequence+100000)
                xy=(random.Random(sequence).randrange(81),random.Random(sequence+9).randrange(81))
                with torch.no_grad():generated=rollout(c);rr=decode_tile(vae,real,xy);rf=decode_tile(vae,generated,xy)
                if sequence%2==0:rr=rr.flip(-1);rf=rf.flip(-1)
                r1_on=dcount%16==15 or (args.preflight and dcount==0)
                rr.requires_grad_(r1_on);pr=disc(rr,c['pooled_prompt_embeds'],return_parts=True);pf=disc(rf,c['pooled_prompt_embeds'],return_parts=True)
                lr=pr['total'];lf=pf['total'];hinge=discriminator_loss(lr,lf);ld=hinge;r1=torch.zeros((),device=dev)
                if r1_on:
                    dr=torch.autograd.grad(lr.sum(),rr,create_graph=True)[0];r1=dr.square().flatten(1).sum(1).mean();ld=ld+.05*16*r1
                d_records.append({'original_index':data['original_indices'][idx],'D_hinge':hinge.item(),'D_total_loss':ld.item(),'R1_raw':r1.item(),'R1_weighted':(.05*16*r1).item(),'R1_active':r1_on,'real_correct':(lr>0).float().mean().item(),'fake_correct':(lf<0).float().mean().item(),'real_scores':{k:pr[k].mean().item() for k in ['total','dino','rgb']},'fake_scores':{k:pf[k].mean().item() for k in ['total','dino','rgb']},'real_dino_heads':pr['dino_heads'][:,0].tolist(),'fake_dino_heads':pf['dino_heads'][:,0].tolist()})
                (ld/accum).backward();del rr,rf,lr,lf,ld,hinge,r1,pr,pf
                if r1_on:del dr
                t=torch.randint(0,1000,(1,),device=dev);eps=torch.randn_like(generated);xt=sched.noise(generated,eps,t)
                loss=(pred(fake,xt,t,c)-eps).square().mean();(loss/accum).backward();losses.append(loss.item())
                assert all(p.grad is None for p in student.parameters());del generated,xt,loss,eps
                timed('fake_D',micro+1)
            average_gradients(disc,world);dn=torch.nn.utils.clip_grad_norm_(disc.trainable(),10,error_if_nonfinite=True).item();od.step();od.zero_grad(set_to_none=True);disc.set_heads(False)
            fn=normclip(fake);of.step();of.zero_grad(set_to_none=True);fcount+=1;dcount+=1
            row={'timestamp':stamp(),'phase':'fake_D','fake_update':fcount,'D_update':dcount,'mean_fake_loss':sum(losses)/len(losses),'fake_grad_norm':fn,'D_grad_norm':dn,'D_records':d_records,'effective_batch':effective}
            log.write(json.dumps(row)+'\n');log.flush();print('FAKE_D_DONE',rank,fcount,flush=True)
            if boundary_actions():stopping=True;break
        if stopping:break
        og.zero_grad(set_to_none=True);records=[]
        for micro in range(accum):
            sequence=gcount*effective+micro*world+rank;idx=order[sequence%n];c=cond(idx);real=data['latents'][idx:idx+1].to(dev,dtype=torch.float32)
            torch.manual_seed(20261017+sequence+200000)
            xy=(random.Random(sequence+100).randrange(81),random.Random(sequence+109).randrange(81))
            generated=rollout(c,stage,True);t=torch.randint(20,920,(1,),device=dev);eps=torch.randn_like(generated);th=torch.randint(940,1000,(1,),device=dev);peps=torch.randn_like(real)
            use_dmd=not(variant=='gan4' and stage==4);use_gan=stage==4
            loss=torch.zeros((),device=dev);row={'phase':'student_micro','g':gcount,'micro':micro,'stage':stage,'variant':variant,'dmd':use_dmd,'gan':use_gan,'original_index':data['original_indices'][idx]}
            dmd_rms=None
            if use_dmd:
                xt=sched.noise(generated,eps,t)
                with torch.no_grad():f0=sched.clean(xt,pred(fake,xt,t,c),t);t0=sched.clean(xt,guide(xt,t,c),t)
                ld,direction=dmd_surrogate(generated,f0,t0);loss=loss+ld;dmd_rms=(2*direction/generated.numel()).square().mean().sqrt();row['dmd_loss']=ld.item()
            if use_gan:
                # Diagnostic DMD reference for GAN-only is never added to loss.
                if not use_dmd and micro==0:
                    with torch.no_grad():
                        probe_xt=sched.noise(generated.detach(),eps,t);probe_fake=sched.clean(probe_xt,pred(fake,probe_xt,t,c),t);probe_teacher=sched.clean(probe_xt,guide(probe_xt,t,c),t)
                        _,probe_direction=dmd_surrogate(generated.detach(),probe_fake,probe_teacher);dmd_rms=(2*probe_direction/generated.numel()).square().mean().sqrt()
                    del probe_xt,probe_fake,probe_teacher,probe_direction
                rgb=decode_tile(vae,generated,xy,True);parts=disc(rgb,c['pooled_prompt_embeds'],return_parts=True);adv=-parts['total'].mean();ag=torch.autograd.grad(adv,generated,retain_graph=True)[0];ar=ag.square().mean().sqrt();assert torch.isfinite(ag).all() and ar>0
                loss=loss+weight*adv;row.update(gan_loss=adv.item(),gan_gradient_rms=ar.item(),gan_weight=weight,gan_dmd_ratio=(weight*ar/dmd_rms).item() if dmd_rms is not None else None)
                row.update(gan_scores={k:parts[k].mean().item() for k in ['total','dino','rgb']},gan_dino_heads=parts['dino_heads'][:,0].tolist(),ratio_is_diagnostic_only=not use_dmd)
            assert loss.requires_grad and torch.isfinite(loss);(loss/accum).backward()
            del generated,loss
            if use_dmd:del xt,f0,t0,ld,direction,dmd_rms
            if use_gan:del rgb,adv,ag,ar,parts
            xh=sched.noise(real,peps,th)
            with torch.no_grad():
                xk=xh
                for k in range(2):tk=th-k*10;xk=sched.jump(xk,guide(xk,tk,c),tk,tk-10)
                target_pcm=sched.jump(xk,pred(student,xk,th-20,c),th-20,919).detach()
            boundary=sched.jump(xh,pred(student,xh,th,c),th,919);pcm=(boundary-target_pcm).square().mean();(10*pcm/accum).backward();row['pcm_loss']=pcm.item();records.append(row)
            del xh,xk,boundary,target_pcm,pcm
            assert all(p.grad is None for p in fake.parameters()) and all(p.grad is None for p in disc.parameters())
            timed('student',micro+1)
        gn=normclip(student);og.step();og.zero_grad(set_to_none=True);gcount+=1
        log.write(json.dumps({'timestamp':stamp(),'phase':'student_done','g':gcount,'grad_norm':gn,'effective_batch':effective,'records':records,'peak_allocated_GiB':torch.cuda.max_memory_allocated()/1024**3})+'\n');log.flush();print('STUDENT_DONE',rank,gcount,gn,flush=True)
        if args.preflight or gcount%max(1,math.ceil(128/effective))==0:evaluate('periodic')
        if boundary_actions():break
    snapshot('preflight_complete' if args.preflight else 'run_end','cycle_boundary',accum)
    if rank==0:atomic(out/'status.json',{'status':'preflight_complete' if args.preflight else 'complete_or_stopped','timestamp':stamp(),'student_updates':gcount,'fake_updates':fcount,'D_updates':dcount,'effective_batch':effective,'peak_allocated_GiB':torch.cuda.max_memory_allocated()/1024**3,'gan_weight':weight})
    print('V3_COMPLETE',rank,flush=True);dist.destroy_process_group()
if __name__=='__main__':main()
