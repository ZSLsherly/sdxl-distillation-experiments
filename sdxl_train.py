"""Four-GPU full-parameter SDXL CrossDistill feasibility experiment."""
import argparse,contextlib,gc,json,os,time,statistics,threading,subprocess
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
    args=ap.parse_args();rank=int(os.environ.get('RANK',0));world=int(os.environ.get('WORLD_SIZE',1));device_id=int(os.environ.get('LOCAL_RANK',0))
    torch.cuda.set_device(device_id);device=torch.device('cuda',device_id)
    if world>1:dist.init_process_group('nccl',timeout=timedelta(minutes=10))
    root=Path(args.root);out=root/'outputs'/args.run;out.mkdir(parents=True,exist_ok=True)
    (out/f'worker_rank{rank}.pid').write_text(str(os.getpid()))
    log=(out/f'rank{rank}.jsonl').open('w');torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=True;torch.manual_seed(20261017)
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
    opt_g=torch.optim.AdamW(student.parameters(),lr=5e-7,betas=(.9,.999),weight_decay=.01,foreach=False)
    opt_f=torch.optim.AdamW(fake.parameters(),lr=1e-7,betas=(.9,.999),weight_decay=.01,foreach=False)
    tracked_g=next(p for p in student.parameters() if local(p).numel()>4096)
    tracked_f=next(p for p in fake.parameters() if local(p).numel()>4096)
    before_g=local(tracked_g).detach().flatten()[:4096].cpu().clone()
    before_f=local(tracked_f).detach().flatten()[:4096].cpu().clone()
    torch.manual_seed(20261017+rank)
    tau=args.crossover;knots=[999,tau,499,249,-1];assert 999>tau>499
    def predict(model,x,t,c,unconditional=False):
        with torch.autocast('cuda',dtype=torch.bfloat16):
            return model(x,t,encoder_hidden_states=c['negative_prompt_embeds'] if unconditional else c['prompt_embeds'],added_cond_kwargs={'text_embeds':c['negative_pooled_prompt_embeds'] if unconditional else c['pooled_prompt_embeds'],'time_ids':c['time_ids']},return_dict=False)[0].float()
    def guided(x,t,c):
        # Sequential CFG keeps inference activation memory bounded.
        e_un=predict(teacher,x,t,c,True);e_cond=predict(teacher,x,t,c)
        return e_un+8.0*(e_cond-e_un)
    def rollout(c,shape,grad=False):
        x=torch.randn(shape,device=device)
        for i,(t,s) in enumerate(zip(knots[:-1],knots[1:])):
            # Same last-step truncated gradient used in the released CrossDistill code.
            with torch.enable_grad() if grad and i==3 else torch.no_grad():
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
        idx=(iteration*world+rank)%len(data['latents'])
        x_real=data['latents'][idx:idx+1].to(device,dtype=torch.float32)
        c={k:data[k][idx:idx+1].to(device) for k in ['prompt_embeds','negative_prompt_embeds','pooled_prompt_embeds','negative_pooled_prompt_embeds','time_ids']}
        c_pcm={k:(data['pcm_'+k] if k!='time_ids' else data[k])[idx:idx+1].to(device) for k in c}
        is_student=iteration%5==4
        opt_g.zero_grad(set_to_none=True);opt_f.zero_grad(set_to_none=True)
        if is_student:
            generated=rollout(c,x_real.shape,grad=True)
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
            (10.0*loss_pcm).backward()
            grad_norm,n_grads=grad_stats(student)
            assert all(p.grad is None for p in fake.parameters())
            opt_g.step();phase='student';loss=loss_dmd+10*loss_pcm
            extras={'dmd_loss':loss_dmd.item(),'pcm_loss':loss_pcm.item(),'pcm_t':th.item(),'dmd_t':t.item()}
        else:
            with torch.no_grad():generated=rollout(c,x_real.shape)
            t=torch.randint(20,tau+1,(1,),device=device);eps=torch.randn_like(generated)
            xt=sched.noise(generated,eps,t)
            loss=(predict(fake,xt,t,c)-eps).square().mean();assert torch.isfinite(loss)
            loss.backward();grad_norm,n_grads=grad_stats(fake)
            assert all(p.grad is None for p in student.parameters())
            opt_f.step();phase='fake_score';extras={'fake_t':t.item()}
        assert all(p.grad is None for p in teacher.parameters())
        torch.cuda.synchronize();elapsed=time.perf_counter()-tick
        row={'iteration':iteration,'phase':phase,'seconds':elapsed,'loss':loss.item(),'grad_norm':grad_norm,'gradient_tensors':n_grads,'peak_allocated_GiB':torch.cuda.max_memory_allocated()/1024**3,'peak_reserved_GiB':torch.cuda.max_memory_reserved()/1024**3,**extras}
        log.write(json.dumps(row)+'\n');log.flush();records.append(row)
        print(f'RANK {rank}: STEP {iteration} {phase} {elapsed:.3f}s loss={loss.item():.6g} peak={row["peak_allocated_GiB"]:.2f}GiB',flush=True)
    if world>1:dist.barrier()
    torch.cuda.synchronize();train_seconds=time.perf_counter()-start
    delta_g=(local(tracked_g).detach().flatten()[:4096].cpu()-before_g).abs().max().item()
    delta_f=(local(tracked_f).detach().flatten()[:4096].cpu()-before_f).abs().max().item()
    assert delta_g>0 and delta_f>0,(delta_g,delta_f)
    summary={'rank':rank,'world_size':world,'parameters_per_unet':count,'resolution':args.resolution,'updates':args.updates,'student_updates':sum(r['phase']=='student' for r in records),'fake_updates':sum(r['phase']=='fake_score' for r in records),'train_seconds':train_seconds,'peak_allocated_GiB':torch.cuda.max_memory_allocated()/1024**3,'peak_reserved_GiB':torch.cuda.max_memory_reserved()/1024**3,'student_max_abs_parameter_delta':delta_g,'fake_max_abs_parameter_delta':delta_f,'config':{'schedule':knots,'crossover_native_timestep':tau,'teacher_cfg':8,'student_cfg':1,'pcm_weight':10,'lr_student':5e-7,'lr_fake':1e-7,'batch_per_gpu':1,'global_batch':world,'precision':'FP32 master + BF16 compute, FP32 gradient reduction','sharding':'FSDP2' if world>1 else 'none','rollout_gradient':'last low-noise step only; relay detached','data':'DMD2 official real LAION-Aesthetics latent shard subset','loss_reduction':'mean latent coordinates','EMA':False},'status':'training_complete'}
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
