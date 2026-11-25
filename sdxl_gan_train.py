"""Controlled SDXL four-knot TD/DMD/PixelGAN feasibility ablation, four GPUs.

Not an official CrossDistill, DMD2, DMDX or SenseFlow reproduction. The student
and score UNets share parameters across timesteps. Gradient truncation isolates
activations, not parameter subspaces. GAN supervises clean estimates, never raw
noisy intermediate states. Warm-start is an earlier 12-update CrossDistill pilot.
"""
import argparse,gc,hashlib,json,os,random,shutil,statistics,subprocess,threading,time
from pathlib import Path
from datetime import timedelta
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed._composable.fsdp import fully_shard,MixedPrecisionPolicy
from diffusers import UNet2DConditionModel,DDIMScheduler,AutoencoderKL
from diffusers.models.attention import BasicTransformerBlock
from diffusers.models.resnet import ResnetBlock2D,Downsample2D,Upsample2D
from diffusers.models.transformers.transformer_2d import Transformer2DModel
from sdxl_objectives import VPSchedule,dmd_surrogate,verify_math
from sdxl_pixel_gan import PixelPatchDiscriminator,decode_tile,discriminator_loss,average_gradients

def local(t):return t.to_local() if hasattr(t,'to_local') else t

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',default='/data5/shenglin.zeng/sdxl_crossdistill')
    ap.add_argument('--run',required=True)
    ap.add_argument('--variant',choices=['dmd','gan4','joint4','joint34'],default='dmd')
    ap.add_argument('--student-updates',type=int,default=24)
    ap.add_argument('--calibrate-only',action='store_true')
    ap.add_argument('--preflight',action='store_true',help='Exercise 3 G / 15 fake+D updates, including R1, without costly validation/export')
    ap.add_argument('--seed',type=int,default=20261017)
    args=ap.parse_args()
    rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);gpu=int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(gpu);device=torch.device('cuda',gpu);torch.set_num_threads(4)
    dist.init_process_group('nccl',timeout=timedelta(minutes=10))
    root=Path(args.root);out=root/'outputs'/args.run;out.mkdir(parents=True,exist_ok=True)
    assert not (out/'summary.json').exists(),'Completed run cannot be overwritten'
    (out/f'worker_rank{rank}.pid').write_text(str(os.getpid()))
    if rank==0:
        (out/'args.json').write_text(json.dumps(vars(args),indent=2))
        for src in [Path(__file__),Path(__file__).with_name('sdxl_pixel_gan.py')]:
            shutil.copyfile(src,out/src.name)
        (out/'source_sha256.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(__file__).with_name('sdxl_pixel_gan.py')]},indent=2))
    base=root/'models/sdxl-base-1.0'
    init=root/'outputs/hparam_pilot/lr5e7_pcm10/export/unet'
    calibration=root/'outputs/gan_stage_pilot/calibration'
    torch.backends.cuda.matmul.allow_tf32=True;verify_math();torch.manual_seed(args.seed)
    data=torch.load(root/'datasets/prepared.pt',weights_only=True,map_location='cpu')
    sched=VPSchedule(DDIMScheduler.from_pretrained(base,subfolder='scheduler',local_files_only=True).alphas_cumprod)
    mesh=init_device_mesh('cuda',(world,))
    policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16,reduce_dtype=torch.float32,cast_forward_inputs=True)
    def load(train,source):
        model=UNet2DConditionModel.from_pretrained(source,torch_dtype=torch.float32 if train else torch.bfloat16,local_files_only=True,**({'variant':'fp16'} if source==base/'unet' else {})).to(device)
        model.requires_grad_(train);model.train(train)
        if train:
            model.enable_gradient_checkpointing()
            for module in reversed(list(model.modules())):
                if isinstance(module,(BasicTransformerBlock,ResnetBlock2D,Transformer2DModel,Downsample2D,Upsample2D)):
                    fully_shard(module,mesh=mesh,mp_policy=policy)
            fully_shard(model,mesh=mesh,mp_policy=policy)
        return model
    print(f'RANK {rank}: LOAD student/fake/teacher/VAE',flush=True)
    student=load(True,init);fake=load(True,base/'unet');teacher=load(False,base/'unet')
    vae=AutoencoderKL.from_pretrained(base,subfolder='vae',variant='fp16',torch_dtype=torch.float32,local_files_only=True).to(device).eval().requires_grad_(False)
    discriminator=PixelPatchDiscriminator().to(device)
    opt_g=torch.optim.AdamW(student.parameters(),lr=5e-7,betas=(.9,.999),weight_decay=.01,foreach=False)
    opt_f=torch.optim.AdamW(fake.parameters(),lr=5e-7,betas=(.9,.999),weight_decay=.01,foreach=False)
    opt_d=torch.optim.AdamW(discriminator.parameters(),lr=1e-4,betas=(0,.99),weight_decay=0)
    if not args.calibrate_only:
        discriminator.load_state_dict(torch.load(calibration/'discriminator.pt',weights_only=True,map_location='cpu'))
        gan_weight=json.loads((calibration/'summary.json').read_text())['gan_weight']
    tracked=next(p for p in student.parameters() if local(p).numel()>4096)
    initial_param=local(tracked).detach().flatten()[:4096].cpu().clone()
    torch.manual_seed(args.seed+rank)
    train_order=random.Random(args.seed+31).sample(range(96),96)
    if rank==0:
        (out/'data_split.json').write_text(json.dumps({'train_cache_indices':train_order,'validation_pool_indices':list(range(96,128)),'evaluated_validation_indices':list(range(112,120)),'reserved_unseen_validation_indices':list(range(120,128)),'conditions':'matching pcm_prompts for all branches; cached independent DMD prompts are not used in this ablation'},indent=2))
    knots=[999,919,499,199,-1]
    keys=['prompt_embeds','negative_prompt_embeds','pooled_prompt_embeds','negative_pooled_prompt_embeds','time_ids']
    def condition(idx):return {k:(data['pcm_'+k] if k!='time_ids' else data[k])[idx:idx+1].to(device) for k in keys}
    def predict(model,x,t,c,un=False):
        with torch.autocast('cuda',dtype=torch.bfloat16):
            return model(x,t,encoder_hidden_states=c['negative_prompt_embeds'] if un else c['prompt_embeds'],added_cond_kwargs={'text_embeds':c['negative_pooled_prompt_embeds'] if un else c['pooled_prompt_embeds'],'time_ids':c['time_ids']},return_dict=False)[0].float()
    def guided(x,t,c):
        eu=predict(teacher,x,t,c,True);ec=predict(teacher,x,t,c)
        return eu+8*(ec-eu)
    def rollout(c,grad=False,stage=4,noise=None):
        x=torch.randn((1,4,128,128),device=device) if noise is None else noise.clone()
        for i in range(stage):
            t=knots[i];s=knots[i+1] if i<stage-1 else -1
            with torch.enable_grad() if grad and i==stage-1 else torch.no_grad():
                tt=torch.tensor([t],device=device)
                x=sched.jump(x,predict(student,x,tt,c),tt,s)
            if i<stage-1:x=x.detach()
        return x
    def grad_norm(model):
        norm=torch.zeros((),device=device,dtype=torch.float64);n=0
        for p in model.parameters():
            if p.grad is not None:
                g=local(p.grad);assert torch.isfinite(g).all();norm+=g.double().square().sum();n+=1
        dist.all_reduce(norm);assert n>0
        value=norm.sqrt().item();factor=min(1,10/(value+1e-12))
        for p in model.parameters():
            if p.grad is not None:p.grad.mul_(factor)
        return value,factor
    def d_step(generated,real,c,xy,number):
        discriminator.requires_grad_(True).train();opt_d.zero_grad(set_to_none=True)
        with torch.no_grad():
            rgb_real=decode_tile(vae,real,xy);rgb_fake=decode_tile(vae,generated.detach(),xy)
        flip=number%2==0
        if flip:rgb_real=rgb_real.flip(-1);rgb_fake=rgb_fake.flip(-1)
        r1_on=number%16==15 or (args.preflight and number==0)
        rgb_real.requires_grad_(r1_on)
        lr=discriminator(rgb_real,c['pooled_prompt_embeds']);lf=discriminator(rgb_fake,c['pooled_prompt_embeds'])
        loss=discriminator_loss(lr,lf);r1=torch.zeros((),device=device)
        if r1_on:
            derivative=torch.autograd.grad(lr.sum(),rgb_real,create_graph=True)[0]
            r1=derivative.square().flatten(1).sum(1).mean()
            loss=loss+(.1/2)*16*r1
        loss.backward();average_gradients(discriminator,world)
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in discriminator.parameters())
        torch.nn.utils.clip_grad_norm_(discriminator.parameters(),10);opt_d.step()
        opt_d.zero_grad(set_to_none=True);discriminator.requires_grad_(False).eval()
        return {'discriminator_loss':loss.item(),'real_logit':lr.mean().item(),'fake_logit':lf.mean().item(),'r1':r1.item()}
    if args.calibrate_only:
        diagnostics=[]
        for number in range(10):
            idx=train_order[number*world+rank];c=condition(idx);real=data['latents'][idx:idx+1].to(device,dtype=torch.float32)
            with torch.no_grad():generated=rollout(c)
            diagnostics.append(d_step(generated,real,c,(40,40),number))
        c=condition(train_order[40+rank])
        with torch.no_grad():
            generated=rollout(c);t=torch.tensor([199],device=device);xt=sched.noise(generated,torch.randn_like(generated),t)
            f0=sched.clean(xt,predict(fake,xt,t,c),t);teacher0=sched.clean(xt,guided(xt,t,c),t)
        generated.requires_grad_(True)
        loss,direction=dmd_surrogate(generated,f0,teacher0)
        adv=-discriminator(decode_tile(vae,generated,(40,40),True),c['pooled_prompt_embeds']).mean()
        g_adv=torch.autograd.grad(adv,generated)[0]
        rms_adv=g_adv.square().mean().sqrt();rms_dmd=(2*direction/generated.numel()).square().mean().sqrt()
        norms=torch.stack([rms_adv,rms_dmd]);dist.all_reduce(norms);norms/=world
        weight=float((.15*norms[1]/norms[0].clamp_min(1e-12)).clamp(.001,100))
        assert torch.isfinite(g_adv).all() and rms_adv>0
        assert all(p.grad is None for p in vae.parameters())
        if rank==0:
            torch.save(discriminator.state_dict(),out/'discriminator.pt')
            (out/'summary.json').write_text(json.dumps({'status':'calibration_complete','gan_weight':weight,'latent_gradient_rms_gan':norms[0].item(),'latent_gradient_rms_dmd':norms[1].item(),'target_initial_gradient_ratio':.15,'discriminator_updates':10,'calibration_images':40,'norm_timesteps':[199],'norm_noise_draw':'fixed RNG after D calibration; one paired caption per rank','caveat':'Fixed weight; ratio will change as discriminator learns. No claim of optimal hyperparameter.','discriminator_log_rank0':diagnostics},indent=2))
        print(f'RANK {rank}: CALIBRATED GAN_WEIGHT {weight:.8g}',flush=True)
        dist.barrier();dist.destroy_process_group();return

    @torch.no_grad()
    def validate(stage):
        rows=[];metrics=[];refs_dir=root/'outputs/gan_stage_pilot/validation_refs_112_119';refs_dir.mkdir(parents=True,exist_ok=True)
        refpath=refs_dir/f'rank{rank}.pt';refs=torch.load(refpath,weights_only=True) if refpath.exists() else None;new=[]
        rng=torch.random.get_rng_state();cuda_rng=torch.cuda.get_rng_state()
        for j in range(2):
            idx=112+rank+j*world;c=condition(idx)
            for seed in [101,102]:
                noise=torch.randn((1,4,128,128),device=device,generator=torch.Generator(device=device).manual_seed(seed))
                if refs is None:
                    x=noise.clone();boundary=None;grid=sorted(set(range(999,-1,-20))|{919,499,199,0},reverse=True)
                    for t,s in zip(grid,grid[1:]+[-1]):
                        tt=torch.tensor([t],device=device);x=sched.jump(x,guided(x,tt,c),tt,s)
                        if s==919:boundary=x.cpu()
                    ref={'cache_index':idx,'seed':seed,'final':x.cpu(),'boundary':boundary};new.append(ref)
                else:
                    ref=refs[len(rows)];assert (ref['cache_index'],ref['seed'])==(idx,seed)
                x=noise.clone();boundary=None;estimates=[]
                for t,s in zip(knots[:-1],knots[1:]):
                    tt=torch.tensor([t],device=device);eps=predict(student,x,tt,c)
                    estimates.append(sched.clean(x,eps,tt).cpu());x=sched.jump(x,eps,tt,s)
                    if s==919:boundary=x.cpu()
                final=x.cpu();assert torch.isfinite(final).all()
                rows.append({'cache_index':idx,'seed':seed,'prompt':data['pcm_prompts'][idx],'final':final,'boundary':boundary,'clean_estimates':estimates})
                metrics.append({'cache_index':idx,'seed':seed,'boundary_relative_mse':((boundary-ref['boundary']).square().mean()/ref['boundary'].square().mean()).item(),'final_teacher_pair_rmse':(final-ref['final']).square().mean().sqrt().item(),'latent_std':final.std().item(),'finite':True})
        if refs is None:torch.save(new,refpath)
        torch.save(rows,out/f'validation_{stage}_rank{rank}.pt')
        (out/f'validation_{stage}_rank{rank}.json').write_text(json.dumps(metrics,indent=2))
        torch.random.set_rng_state(rng);torch.cuda.set_rng_state(cuda_rng)
        print(f'RANK {rank}: VALIDATION_{stage}',flush=True)
    if not args.preflight:validate('before')
    stop=threading.Event();nv=[0]*world;nv_total=[0]*world
    def monitor():
        while not stop.is_set():
            try:
                pids={int((out/f'worker_rank{i}.pid').read_text()):i for i in range(world)}
                lines=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True).splitlines()
                for line in lines:
                    pid,mem=map(int,line.split(','))
                    if pid in pids:nv[pids[pid]]=max(nv[pids[pid]],mem)
                lines=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).splitlines()
                for i,line in enumerate(lines[:world]):nv_total[i]=max(nv_total[i],int(line))
            except Exception:pass
            stop.wait(1)
    if rank==0:threading.Thread(target=monitor,daemon=True).start()
    dist.barrier();torch.cuda.reset_peak_memory_stats();start=time.perf_counter();records=[]
    log=(out/f'rank{rank}.jsonl').open('w')
    for iteration in range(args.student_updates*6):
        dist.barrier();torch.cuda.synchronize();tick=time.perf_counter()
        is_g=iteration%6==5;g_step=iteration//6
        # Separate counters avoid the previous stride aliasing of student samples.
        index_counter=g_step if is_g else (g_step*5+iteration%6)
        idx=train_order[(index_counter*world+rank)%96]
        c=condition(idx);real=data['latents'][idx:idx+1].to(device,dtype=torch.float32)
        crop_rng=random.Random(args.seed+iteration*world+rank+80000);xy=(crop_rng.randrange(0,81),crop_rng.randrange(0,81))
        opt_g.zero_grad(set_to_none=True);opt_f.zero_grad(set_to_none=True)
        extras={}
        if not is_g:
            with torch.no_grad():generated=rollout(c)
            extras.update(d_step(generated,real,c,xy,g_step*5+iteration%6))
            t=torch.randint(0,1000,(1,),device=device);eps=torch.randn_like(generated);xt=sched.noise(generated,eps,t)
            loss=(predict(fake,xt,t,c)-eps).square().mean();loss.backward();norm,clip=grad_norm(fake);opt_f.step();phase='fake_score+pixel_discriminator'
            assert all(p.grad is None for p in student.parameters())
        else:
            stage=2+g_step%3;generated=rollout(c,True,stage)
            # Draw all stochastic targets in every variant, including GAN-only,
            # so dropping DMD4 does not change subsequent noise/data randomness.
            t=torch.randint(20,920,(1,),device=device);eps=torch.randn_like(generated)
            th=torch.randint(940,1000,(1,),device=device);pcm_eps=torch.randn_like(real)
            use_dmd=not(args.variant=='gan4' and stage==4)
            use_gan=(stage==4 and args.variant!='dmd') or (stage==3 and args.variant=='joint34')
            loss=torch.zeros((),device=device);extras={'generator_stage':stage,'dmd_active':use_dmd,'gan_active':use_gan}
            if use_dmd:
                xt=sched.noise(generated,eps,t)
                with torch.no_grad():
                    f0=sched.clean(xt,predict(fake,xt,t,c),t);teacher0=sched.clean(xt,guided(xt,t,c),t)
                ld,direction=dmd_surrogate(generated,f0,teacher0);loss=loss+ld
                extras.update(dmd_loss=ld.item(),dmd_t=t.item(),dmd_latent_gradient_rms=(2*direction/generated.numel()).square().mean().sqrt().item())
            if use_gan:
                rgb=decode_tile(vae,generated,xy,True)
                adv=-discriminator(rgb,c['pooled_prompt_embeds']).mean()
                adv_gradient=torch.autograd.grad(adv,generated,retain_graph=True)[0]
                assert torch.isfinite(adv_gradient).all() and adv_gradient.abs().max()>0,'PixelGAN gradient did not reach SDXL output'
                loss=loss+gan_weight*adv
                extras.update(gan_loss=adv.item(),gan_weight=gan_weight,gan_latent_gradient_rms=adv_gradient.square().mean().sqrt().item(),rgb_absmax=rgb.detach().abs().max().item())
                del rgb,adv_gradient
            assert loss.requires_grad and torch.isfinite(loss)
            loss.backward();low_loss=loss.item()
            del loss,generated
            if use_dmd:del xt,f0,teacher0,direction,ld
            if use_gan:del adv
            # High-noise PCM uses two teacher DDIM substeps; student target stopped.
            xh=sched.noise(real,pcm_eps,th)
            with torch.no_grad():
                xk=xh
                for k in range(2):
                    tk=th-k*10;xk=sched.jump(xk,guided(xk,tk,c),tk,tk-10)
                target=sched.jump(xk,predict(student,xk,th-20,c),th-20,919).detach()
            boundary=sched.jump(xh,predict(student,xh,th,c),th,919)
            pcm=(boundary-target).square().mean();(10*pcm).backward()
            norm,clip=grad_norm(student);opt_g.step();phase='student'
            extras['pcm_loss']=pcm.item();extras['pcm_t']=th.item()
            loss=torch.tensor(low_loss+10*pcm.item(),device=device)
            assert all(p.grad is None for p in fake.parameters())
            assert all(p.grad is None for p in discriminator.parameters())
        assert all(p.grad is None for p in teacher.parameters()) and all(p.grad is None for p in vae.parameters())
        torch.cuda.synchronize();elapsed=time.perf_counter()-tick
        row={'iteration':iteration,'phase':phase,'cache_index':idx,'seconds':elapsed,'loss':loss.item(),'grad_norm':norm,'clip_scale':clip,'peak_allocated_GiB':torch.cuda.max_memory_allocated()/1024**3,'peak_reserved_GiB':torch.cuda.max_memory_reserved()/1024**3,**extras}
        assert torch.isfinite(loss)
        log.write(json.dumps(row)+'\n');log.flush();records.append(row)
        print(f'RANK {rank}: STEP {iteration} {phase} {elapsed:.2f}s loss={loss.item():.6g} peak={row["peak_allocated_GiB"]:.2f}GiB',flush=True)
    dist.barrier();train_seconds=time.perf_counter()-start
    if not args.preflight:validate('after')
    delta=(local(tracked).detach().flatten()[:4096].cpu()-initial_param).abs().max().item();assert delta>0
    summary={'status':'training_complete','rank':rank,'world_size':world,'variant':args.variant,'student_updates':args.student_updates,'fake_updates':args.student_updates*5,'pixel_discriminator_updates':args.student_updates*5,'train_seconds':train_seconds,'peak_allocated_GiB':torch.cuda.max_memory_allocated()/1024**3,'peak_reserved_GiB':torch.cuda.max_memory_reserved()/1024**3,'student_parameter_delta':delta,'config':{'schedule':knots,'init':str(init),'init_training_updates':12,'student_lr':5e-7,'fake_lr':5e-7,'discriminator_lr':1e-4,'student_fake_adamw_betas':[.9,.999],'discriminator_adamw_betas':[0,.99],'discriminator_weight_decay':0,'r1_gamma':.1,'r1_interval':16,'pcm_weight':10,'teacher_cfg':8,'gan_weight':gan_weight,'gan_weight_method':'common initial RGB-to-latent gradient calibration, target 0.15 DMD, fixed thereafter','training_resolution':1024,'discriminator_crop_rgb':256,'vae_decode_tile_rgb':384,'batch_per_gpu':1,'global_batch':world,'precision':'FP32 master/BF16 UNets; FP32 VAE, RGB discriminator','step_supervision':'round-robin stages 2,3,4; each G update also high-noise PCM; truncated last-node gradient','real_data':'VAE reconstructions of LAION-Aesthetics, matching real captions','shared_parameters_warning':'detached prefixes do not freeze earlier-timestep behavior','pixel_discriminator':'from-scratch conditional RGB PatchGAN; not pretrained SAM/DINO','student_unique_images_rank':len({r['cache_index'] for r in records if r['phase']=='student'}),'seed':args.seed,'EMA':False}}
    (out/f'summary_rank{rank}.json').write_text(json.dumps(summary,indent=2))
    if not args.preflight:
        torch.save({k:local(v).detach().cpu().to(torch.bfloat16) for k,v in student.state_dict().items()},out/f'student_rank{rank}.pt')
        (out/f'student_rank{rank}_shapes.json').write_text(json.dumps({k:{'shape':list(v.shape),'placements':[str(p) for p in v.placements] if hasattr(v,'placements') else []} for k,v in student.state_dict().items()},indent=2))
    else:
        summary['status']='preflight_complete'
        summary['preflight_extra_R1_at_first_D_update']=True
    dist.barrier();stop.set()
    if rank==0:
        if not args.preflight:torch.save(discriminator.state_dict(),out/'discriminator.pt')
        summary['nvidia_smi_peak_own_process_MiB_per_gpu']=nv;summary['nvidia_smi_peak_total_MiB_per_gpu']=nv_total
        (out/'summary.json').write_text(json.dumps(summary,indent=2))
    log.close();print(f'RANK {rank}: TRAIN_COMPLETE',flush=True);dist.destroy_process_group()

if __name__=='__main__':main()
