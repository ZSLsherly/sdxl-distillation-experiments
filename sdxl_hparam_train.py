"""Four-GPU full-parameter SDXL CrossDistill feasibility experiment."""
import argparse,contextlib,gc,json,os,time,statistics,threading,subprocess,random,hashlib,shutil
from pathlib import Path
from datetime import timedelta
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed._composable.fsdp import fully_shard,MixedPrecisionPolicy
from diffusers import UNet2DConditionModel,DDIMScheduler
from diffusers.models.attention import BasicTransformerBlock
from diffusers.models.resnet import ResnetBlock2D,Downsample2D,Upsample2D
from diffusers.models.transformers.transformer_2d import Transformer2DModel
from sdxl_objectives import VPSchedule,dmd_surrogate,verify_math

def local(t):return t.to_local() if hasattr(t,'to_local') else t

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',default='/data5/shenglin.zeng/sdxl_crossdistill')
    ap.add_argument('--updates',type=int,default=20);ap.add_argument('--resolution',type=int,default=1024)
    ap.add_argument('--run',default='smoke_1024');ap.add_argument('--crossover',type=int,default=749)
    ap.add_argument('--lr-student',type=float,default=5e-7);ap.add_argument('--lr-fake',type=float,default=5e-7)
    ap.add_argument('--pcm-weight',type=float,default=10);ap.add_argument('--fake-ratio',type=int,default=5)
    ap.add_argument('--max-grad-norm',type=float,default=10);ap.add_argument('--seed',type=int,default=20261017)
    ap.add_argument('--teacher-cfg',type=float,default=8);ap.add_argument('--train-samples',type=int,default=96)
    args=ap.parse_args();rank=int(os.environ.get('RANK',0));world=int(os.environ.get('WORLD_SIZE',1));device_id=int(os.environ.get('LOCAL_RANK',0))
    # Prioritize the user-requested teacher geometry profile between trials.
    # No CUDA context or process group is allocated while waiting.
    if args.run=='hparam_pilot/lr25e8_pcm1':
        marker=Path(args.root)/'logs/teacher_turning.exit'
        for _ in range(240):
            if marker.exists():
                assert marker.read_text().strip()=='0','Teacher turning profile failed'
                break
            time.sleep(5)
        else:raise TimeoutError('Teacher turning profile did not finish')
    torch.cuda.set_device(device_id);device=torch.device('cuda',device_id)
    if world>1:dist.init_process_group('nccl',timeout=timedelta(minutes=10))
    root=Path(args.root);out=root/'outputs'/args.run;out.mkdir(parents=True,exist_ok=True)
    assert not (out/'summary.json').exists(),'Refusing to overwrite a completed trial'
    if rank==0:
        (out/'args.json').write_text(json.dumps(vars(args),indent=2))
        shutil.copyfile(__file__,out/'training_source.py')
        (out/'training_source.sha256').write_text(hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/f'worker_rank{rank}.pid').write_text(str(os.getpid()))
    log=(out/f'rank{rank}.jsonl').open('w');torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=True;torch.manual_seed(args.seed)
    base=root/'models/sdxl-base-1.0';verify_math()
    print(f'RANK {rank}: loading real LAION cache',flush=True)
    data=torch.load(root/'datasets/prepared.pt',map_location='cpu',weights_only=True)
    assert data['resolution']==args.resolution
    sched=VPSchedule(DDIMScheduler.from_pretrained(base,subfolder='scheduler',local_files_only=True).alphas_cumprod)
    mesh=init_device_mesh('cuda',(world,)) if world>1 else None
    policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16,reduce_dtype=torch.float32,cast_forward_inputs=True)
    def load(train):
        model=UNet2DConditionModel.from_pretrained(base,subfolder='unet',variant='fp16',torch_dtype=torch.float32 if train else torch.bfloat16,local_files_only=True)
        model.requires_grad_(train);model.train(train);model.to(device)
        if train:
            model.enable_gradient_checkpointing()
            if world>1:
                for module in reversed(list(model.modules())):
                    if isinstance(module,(BasicTransformerBlock,ResnetBlock2D,Transformer2DModel,Downsample2D,Upsample2D)):
                        fully_shard(module,mesh=mesh,mp_policy=policy)
                fully_shard(model,mesh=mesh,mp_policy=policy)
        return model
    print(f'RANK {rank}: loading student',flush=True);student=load(True)
    print(f'RANK {rank}: loading fake score',flush=True);fake=load(True)
    print(f'RANK {rank}: loading frozen teacher',flush=True);teacher=load(False)
    count=sum(p.numel() for p in student.parameters())
    opt_g=torch.optim.AdamW(student.parameters(),lr=args.lr_student,betas=(.9,.999),weight_decay=.01,foreach=False)
    opt_f=torch.optim.AdamW(fake.parameters(),lr=args.lr_fake,betas=(.9,.999),weight_decay=.01,foreach=False)
    tracked_g=next(p for p in student.parameters() if local(p).numel()>4096)
    tracked_f=next(p for p in fake.parameters() if local(p).numel()>4096)
    before_g=local(tracked_g).detach().flatten()[:4096].cpu().clone()
    before_f=local(tracked_f).detach().flatten()[:4096].cpu().clone()
    torch.manual_seed(args.seed+rank)
    length_rng=random.Random(args.seed)
    tau=args.crossover;knots=[999,tau,int(2*tau/3),int(tau/3),-1];assert 978>=tau>40
    assert args.fake_ratio>0 and args.train_samples<=len(data["latents"])
    train_order=random.Random(args.seed+31).sample(range(args.train_samples),args.train_samples)
    if rank==0:(out/"data_split.json").write_text(json.dumps({"train_cache_indices":train_order,"validation_cache_indices":list(range(args.train_samples,len(data["latents"])))},indent=2))
    def predict(model,x,t,c,unconditional=False):
        with torch.autocast('cuda',dtype=torch.bfloat16):
            return model(x,t,encoder_hidden_states=c['negative_prompt_embeds'] if unconditional else c['prompt_embeds'],added_cond_kwargs={'text_embeds':c['negative_pooled_prompt_embeds'] if unconditional else c['pooled_prompt_embeds'],'time_ids':c['time_ids']},return_dict=False)[0].float()
    def guided(x,t,c):
        # Sequential CFG keeps inference activation memory bounded.
        e_un=predict(teacher,x,t,c,True);e_cond=predict(teacher,x,t,c)
        return e_un+args.teacher_cfg*(e_cond-e_un)
    def rollout(c,shape,grad=False,length=4):
        x=torch.randn(shape,device=device)
        active=knots[:length]+[-1]
        for i,(t,s) in enumerate(zip(active[:-1],active[1:])):
            # Same last-step truncated gradient used in the released CrossDistill code.
            with torch.enable_grad() if grad and i==length-1 else torch.no_grad():
                tt=torch.full((shape[0],),t,device=device,dtype=torch.long)
                x=sched.jump(x,predict(student,x,tt,c),tt,s)
            if i==0:
                x=x.detach();assert not x.requires_grad
        return x
    def grad_stats(model):
        norm=torch.zeros((),device=device,dtype=torch.float64);finite=torch.ones((),device=device,dtype=torch.int32);n=0
        for p in model.parameters():
            if p.grad is not None:
                g=local(p.grad);norm+=g.double().square().sum();finite*=torch.isfinite(g).all().int();n+=1
        if world>1:dist.all_reduce(norm);dist.all_reduce(finite,op=dist.ReduceOp.MIN)
        assert finite.item()==1 and n>0,'non-finite or missing gradients'
        return norm.sqrt().item(),n

    @torch.no_grad()
    def validate(stage):
        # A fixed, disjoint LAION holdout; COCO is not used to rank trials.
        # Teacher-relative final error is a diagnostic, not a DMD quality metric.
        refs_dir=root/'outputs/hparam_pilot/validation_refs';refs_dir.mkdir(parents=True,exist_ok=True)
        refpath=refs_dir/f'tau{tau}_rank{rank}.pt'
        rows=[];images=[];cached=[]
        refs=torch.load(refpath,weights_only=True) if refpath.exists() else None
        cases=[(args.train_samples+rank+j*world,seed) for j in range(2) for seed in [101,102]]
        rng=torch.random.get_rng_state();cuda_rng=torch.cuda.get_rng_state()
        for case_id,(idx,seed) in enumerate(cases):
            assert idx<len(data['latents'])
            c={k:data[k][idx:idx+1].to(device) for k in ['prompt_embeds','negative_prompt_embeds','pooled_prompt_embeds','negative_pooled_prompt_embeds','time_ids']}
            noise=torch.randn((1,4,128,128),device=device,generator=torch.Generator(device=device).manual_seed(seed))
            if refs is None:
                xt=noise.clone();grid=sorted(set(range(999,-1,-20))|{tau,0},reverse=True);tb=None
                for t,s in zip(grid,grid[1:]+[-1]):
                    tt=torch.tensor([t],device=device,dtype=torch.long)
                    xt=sched.jump(xt,guided(xt,tt,c),tt,s)
                    if s==tau:tb=xt.cpu()
                assert tb is not None
                ref={'boundary':tb,'final':xt.cpu(),'cache_index':idx,'seed':seed}
                cached.append(ref)
            else:
                ref=refs[case_id];assert ref['cache_index']==idx and ref['seed']==seed
            x=noise.clone();boundary=None
            for t,s in zip(knots[:-1],knots[1:]):
                tt=torch.tensor([t],device=device,dtype=torch.long)
                x=sched.jump(x,predict(student,x,tt,c),tt,s)
                if s==tau:boundary=x.cpu()
            x=x.cpu();target=ref['boundary']
            rows.append({'cache_index':idx,'seed':seed,'boundary_mse':(boundary-target).square().mean().item(),
                         'boundary_relative_mse':((boundary-target).square().mean()/target.square().mean().clamp_min(1e-8)).item(),
                         'final_teacher_pair_rmse':(x-ref['final']).square().mean().sqrt().item(),
                         'latent_mean':x.mean().item(),'latent_std':x.std().item(),'finite':bool(torch.isfinite(x).all())})
            images.append({'cache_index':idx,'seed':seed,'prompt':data['prompts'][idx],'boundary':boundary,'final':x})
        if refs is None:torch.save(cached,refpath)
        assert all(r['finite'] for r in rows)
        torch.save(images,out/f'validation_{stage}_rank{rank}.pt')
        (out/f'validation_{stage}_rank{rank}.json').write_text(json.dumps(rows,indent=2))
        torch.random.set_rng_state(rng);torch.cuda.set_rng_state(cuda_rng)
        print(f'RANK {rank}: VALIDATION_{stage.upper()} boundary_relative_mse={statistics.mean(r["boundary_relative_mse"] for r in rows):.6f}',flush=True)

    validate('before')
    stop=threading.Event();nv_peaks=[0]*world;nv_process_peaks=[0]*world
    def monitor():
        while not stop.is_set():
            try:
                lines=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).splitlines()
                for i,line in enumerate(lines[:world]):nv_peaks[i]=max(nv_peaks[i],int(line))
                pid_to_rank={int((out/f'worker_rank{i}.pid').read_text()):i for i in range(world)}
                processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True).splitlines()
                for line in processes:
                    pid,mem=map(int,line.split(','))
                    if pid in pid_to_rank:
                        i=pid_to_rank[pid];nv_process_peaks[i]=max(nv_process_peaks[i],mem)
            except Exception:pass
            stop.wait(.5)
    if rank==0:threading.Thread(target=monitor,daemon=True).start()
    if world>1:dist.barrier()
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter();records=[]
    print(f'RANK {rank}: TRAIN_BEGIN parameters={count} resolution={args.resolution}',flush=True)
    for iteration in range(args.updates):
        if world>1:dist.barrier()
        torch.cuda.synchronize();tick=time.perf_counter()
        idx=train_order[(iteration*world+rank)%len(train_order)]
        x_real=data['latents'][idx:idx+1].to(device,dtype=torch.float32)
        c={k:data[k][idx:idx+1].to(device) for k in ['prompt_embeds','negative_prompt_embeds','pooled_prompt_embeds','negative_pooled_prompt_embeds','time_ids']}
        c_pcm={k:(data['pcm_'+k] if k!='time_ids' else data[k])[idx:idx+1].to(device) for k in c}
        is_student=iteration%(args.fake_ratio+1)==args.fake_ratio
        opt_g.zero_grad(set_to_none=True);opt_f.zero_grad(set_to_none=True)
        if is_student:
            rollout_length=length_rng.choice([2,3,4])
            generated=rollout(c,x_real.shape,grad=True,length=rollout_length)
            t=torch.randint(20,tau+1,(1,),device=device);eps=torch.randn_like(generated)
            xt=sched.noise(generated,eps,t)
            with torch.no_grad():
                f0=sched.clean(xt,predict(fake,xt,t,c),t)
                teacher0=sched.clean(xt,guided(xt,t,c),t)
            loss_dmd,direction=dmd_surrogate(generated,f0,teacher0)
            assert torch.isfinite(loss_dmd)
            loss_dmd.backward()
            # Free the low-noise graph before high-noise target forwards.
            del generated,xt,f0,teacher0,direction
            th=torch.randint(tau+21,1000,(1,),device=device)
            eps=torch.randn_like(x_real);xh=sched.noise(x_real,eps,th)
            with torch.no_grad():
                xk=xh
                for k in range(2):
                    tk=th-k*10;tn=tk-10
                    xk=sched.jump(xk,guided(xk,tk,c_pcm),tk,tn)
                target=sched.jump(xk,predict(student,xk,th-20,c_pcm),th-20,tau).detach()
            boundary=sched.jump(xh,predict(student,xh,th,c_pcm),th,tau)
            loss_pcm=(boundary-target).square().mean()
            assert torch.isfinite(loss_pcm)
            (args.pcm_weight*loss_pcm).backward()
            grad_norm,n_grads=grad_stats(student)
            assert all(p.grad is None for p in fake.parameters())
            clip_scale=min(1.0,args.max_grad_norm/(grad_norm+1e-12))
            for parameter in student.parameters():
                if parameter.grad is not None:parameter.grad.mul_(clip_scale)
            opt_g.step();phase='student';loss=loss_dmd+args.pcm_weight*loss_pcm
            extras={'dmd_loss':loss_dmd.item(),'pcm_loss':loss_pcm.item(),'pcm_t':th.item(),'dmd_t':t.item(),'rollout_length':rollout_length,'clip_scale':clip_scale}
        else:
            with torch.no_grad():generated=rollout(c,x_real.shape)
            t=torch.randint(0,1000,(1,),device=device);eps=torch.randn_like(generated)
            xt=sched.noise(generated,eps,t)
            loss=(predict(fake,xt,t,c)-eps).square().mean();assert torch.isfinite(loss)
            loss.backward();grad_norm,n_grads=grad_stats(fake)
            assert all(p.grad is None for p in student.parameters())
            clip_scale=min(1.0,args.max_grad_norm/(grad_norm+1e-12))
            for parameter in fake.parameters():
                if parameter.grad is not None:parameter.grad.mul_(clip_scale)
            opt_f.step();phase='fake_score';extras={'fake_t':t.item(),'clip_scale':clip_scale}
        assert all(p.grad is None for p in teacher.parameters())
        torch.cuda.synchronize();elapsed=time.perf_counter()-tick
        row={'iteration':iteration,'phase':phase,'seconds':elapsed,'loss':loss.item(),'grad_norm':grad_norm,'gradient_tensors':n_grads,'peak_allocated_GiB':torch.cuda.max_memory_allocated()/1024**3,'peak_reserved_GiB':torch.cuda.max_memory_reserved()/1024**3,**extras}
        log.write(json.dumps(row)+'\n');log.flush();records.append(row)
        print(f'RANK {rank}: STEP {iteration} {phase} {elapsed:.3f}s loss={loss.item():.6g} peak={row["peak_allocated_GiB"]:.2f}GiB',flush=True)
    if world>1:dist.barrier()
    torch.cuda.synchronize();train_seconds=time.perf_counter()-start
    validate('after')
    delta_g=(local(tracked_g).detach().flatten()[:4096].cpu()-before_g).abs().max().item()
    delta_f=(local(tracked_f).detach().flatten()[:4096].cpu()-before_f).abs().max().item()
    assert delta_g>0 and delta_f>0,(delta_g,delta_f)
    summary={'rank':rank,'world_size':world,'parameters_per_unet':count,'resolution':args.resolution,'updates':args.updates,'student_updates':sum(r['phase']=='student' for r in records),'fake_updates':sum(r['phase']=='fake_score' for r in records),'train_seconds':train_seconds,'peak_allocated_GiB':torch.cuda.max_memory_allocated()/1024**3,'peak_reserved_GiB':torch.cuda.max_memory_reserved()/1024**3,'student_max_abs_parameter_delta':delta_g,'fake_max_abs_parameter_delta':delta_f,'config':{'schedule':knots,'crossover_native_timestep':tau,'teacher_cfg':args.teacher_cfg,'student_cfg':1,'pcm_weight':args.pcm_weight,'lr_student':args.lr_student,'lr_fake':args.lr_fake,'fake_to_student_ratio':args.fake_ratio,'max_grad_norm':args.max_grad_norm,'train_cache_samples':args.train_samples,'validation_cache_samples':len(data['latents'])-args.train_samples,'seed':args.seed,'batch_per_gpu':1,'global_batch':world,'precision':'FP32 master + BF16 compute, FP32 gradient reduction','sharding':'FSDP2' if world>1 else 'none','rollout_gradient':'uniform rollout length m in {2,3,4}, last low-noise step only; relay detached','data':'DMD2 official real LAION-Aesthetics latent shard subset','loss_reduction':'mean latent coordinates','EMA':False},'status':'training_complete'}
    (out/f'summary_rank{rank}.json').write_text(json.dumps(summary,indent=2))
    # Each rank saves its own model-only shard. No multi-GB gather is required.
    shard={k:local(v).detach().cpu().to(torch.bfloat16) for k,v in student.state_dict().items()}
    torch.save(shard,out/f'student_rank{rank}.pt')
    (out/f'student_rank{rank}_shapes.json').write_text(json.dumps({k:{'shape':list(v.shape),'placements':[str(p) for p in v.placements] if hasattr(v,'placements') else []} for k,v in student.state_dict().items()},indent=2))
    if world>1:dist.barrier()
    stop.set()
    if rank==0:
        summary['nvidia_smi_peak_total_MiB_per_gpu']=nv_peaks
        summary['nvidia_smi_peak_own_process_MiB_per_gpu']=nv_process_peaks
        (out/'summary.json').write_text(json.dumps(summary,indent=2))
    log.close();print(f'RANK {rank}: TRAIN_COMPLETE',flush=True)
    if world>1:dist.destroy_process_group()

if __name__=='__main__':main()
