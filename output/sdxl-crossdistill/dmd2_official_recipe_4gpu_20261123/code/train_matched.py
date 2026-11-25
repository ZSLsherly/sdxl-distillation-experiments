"""Matched 10K runner: upstream DMD2 methods, PyTorch FSDP2, cached text/real latents."""
import argparse,hashlib,json,math,os,random,sys,time
from pathlib import Path
from datetime import timedelta
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).parent/'vendor'))
import torch
import torch.distributed as dist
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed._composable.fsdp import fully_shard,MixedPrecisionPolicy
from torch.distributed.checkpoint.state_dict import get_model_state_dict,StateDictOptions
from diffusers import UNet2DConditionModel
from diffusers.models.attention import BasicTransformerBlock
from diffusers.models.resnet import ResnetBlock2D,Downsample2D,Upsample2D
from diffusers.models.transformers.transformer_2d import Transformer2DModel
import main.sd_unified_model as upstream
from objectives import VPSchedule

class CachedText(torch.nn.Module):
    def __init__(self,args,accelerator):super().__init__()
    def forward(self,batch):return batch['prompt_embeds'],batch['pooled_prompt_embeds']

def shard(net,mesh,policy):
    types=(BasicTransformerBlock,ResnetBlock2D,Transformer2DModel,Downsample2D,Upsample2D)
    for child in reversed(list(net.modules())):
        if isinstance(child,types):fully_shard(child,mesh=mesh,mp_policy=policy)
    fully_shard(net,mesh=mesh,mp_policy=policy)

def seed_for(seed,role,step,rank):
    return int.from_bytes(hashlib.sha256(f'{seed}:{role}:{step}:{rank}'.encode()).digest()[:8],'little')%(2**63-1)

def local(x):return x.to_local() if hasattr(x,'to_local') else x

def clip(params,limit):
    grads=[local(p.grad) for p in params if p.grad is not None]
    if not grads:raise RuntimeError('No gradients')
    total=torch.stack(torch._foreach_norm(grads)).double().square().sum()
    dist.all_reduce(total);norm=total.sqrt().item()
    if not math.isfinite(norm):raise FloatingPointError('Non-finite gradient')
    torch._foreach_mul_(grads,min(1.,limit/(norm+1e-12)))
    return norm

def write(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--run',required=True)
    parser.add_argument('--micro',type=int,default=16);parser.add_argument('--updates',type=int,default=160)
    parser.add_argument('--smoke',action='store_true');a=parser.parse_args()
    cfg=json.loads((a.root/'config.json').read_text());seed=cfg['seed']
    # Official train_sd.py enables TF32. FP32 parameter/activation dtype does
    # not mean disabling the author's TensorFloat-32 matrix/conv kernels.
    torch.backends.cuda.matmul.allow_tf32=True
    torch.backends.cudnn.allow_tf32=True
    accelerator=Accelerator(mixed_precision='no',device_placement=False,
        kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(minutes=20))])
    device=accelerator.device;rank=accelerator.process_index;world=accelerator.num_processes
    assert world in (2,4) and 64%(world*a.micro)==0
    accum=64//(world*a.micro);mesh=init_device_mesh('cuda',(world,))
    torch.cuda.set_device(device);torch.set_num_threads(2);torch.manual_seed(seed)
    out=a.root/'runs'/a.run;out.mkdir(parents=True,exist_ok=True)
    if (out/f'rank{rank}.jsonl').exists():raise FileExistsError('Use fresh output directory')
    data_path=Path(cfg['data_dir']);data=torch.load(data_path/'prepared.pt',map_location='cpu',weights_only=True,mmap=True)
    expected=json.loads((a.root/'code/expected_train_indices.json').read_text())
    assert data['train_count']==10000 and data['original_indices'][:10000]==expected
    assert data['negative_prompt_embeds'].abs().max()==0 and data['negative_pooled_prompt_embeds'].abs().max()==0
    assert data['time_ids'].tolist()==[[1024,1024,0,0,1024,1024]]
    args=SimpleNamespace(model_id=cfg['model_id'],sdxl=True,initialie_generator=True,
        generator_lora=False,grid_size=2,conditioning_timestep=999,use_fp16=True,
        gradient_checkpointing=True,backward_simulation=True,cls_on_clean_image=True,
        gen_cls_loss=True,denoising=True,denoising_timestep=1000,num_denoising_step=4,
        tiny_vae=False,resolution=1024,gan_alone=False,num_train_timesteps=1000,
        min_step_percent=.02,max_step_percent=.979,real_guidance_scale=8.,fake_guidance_scale=1.,
        diffusion_gan=True,diffusion_gan_max_timestep=1000,revision=None)
    upstream.SDXLTextEncoder=CachedText
    # The host user.slice has memory.high=100GiB shared across users.
    # Initialize and shard one rank at a time, avoiding four full CPU copies.
    import gc,ctypes
    for initializing_rank in range(world):
        if rank==initializing_rank:
            print('INIT_RANK_BEGIN',rank,flush=True)
            model=upstream.SDUniModel(args,accelerator)
            model.feedforward_model.to(device);model.guidance_model.to(device)
            # HF non-reentrant activation checkpointing + the project's proven FSDP2 wrappers.
            model.guidance_model.fake_unet.enable_gradient_checkpointing()
            # Keep FP32 fake-score/rollout; upstream owns selective autocast.
            policy=MixedPrecisionPolicy(param_dtype=None,reduce_dtype=torch.float32,cast_forward_inputs=False)
            shard(model.feedforward_model,mesh,policy)
            shard(model.guidance_model.fake_unet,mesh,policy)
            shard(model.guidance_model.real_unet,mesh,policy)
            # The parent shards only its remaining native discriminator head parameters.
            fully_shard(model.guidance_model,mesh=mesh,mp_policy=MixedPrecisionPolicy(
                param_dtype=torch.float32,reduce_dtype=torch.float32,cast_forward_inputs=False))
            gc.collect()
            ctypes.CDLL('libc.so.6').malloc_trim(0)
            print('INIT_RANK_SHARDED',rank,flush=True)
        dist.barrier()
    student_params=list(model.feedforward_model.parameters())
    fake_params=list(model.guidance_model.fake_unet.parameters())+list(model.guidance_model.cls_pred_branch.parameters())
    adam=dict(betas=tuple(cfg['betas']),weight_decay=cfg['weight_decay'],foreach=False)
    og=torch.optim.AdamW(student_params,lr=cfg['student_lr'],**adam);of=torch.optim.AdamW(fake_params,lr=cfg['fake_lr'],**adam)
    orders={role:random.Random(seed+offset).sample(range(10000),10000) for role,offset in [('student',0),('fake',100),('real',200)]}
    schedule=VPSchedule(model.noise_scheduler.alphas_cumprod.to(device))
    manifest=json.loads((data_path/'manifest.json').read_text())
    def batch(role,step,micro):
        pos=[step*64+(micro*a.micro+j)*world+rank for j in range(a.micro)]
        ids=[orders[role][p%10000] for p in pos];ix=torch.tensor(ids)
        return {k:data[src].index_select(0,ix).to(device,dtype=torch.float32) for k,src in
                [('images','latents'),('prompt_embeds','prompt_embeds'),('pooled_prompt_embeds','pooled_prompt_embeds')]}
    def metrics_reduce(values):
        t=torch.tensor(values,device=device,dtype=torch.float64);dist.all_reduce(t);return (t/world).tolist()
    log=(out/f'rank{rank}.jsonl').open('w');g=f=0
    def record(value):
        value.update(rank=rank,time=time.time());log.write(json.dumps(value)+'\n');log.flush()
    precision_seen={}
    def precision_hook(role):
        def hook(module, inputs, output):
            amp=torch.is_autocast_enabled()
            key=f'{role}:autocast={amp}:grad={torch.is_grad_enabled()}'
            value=dict(input=str(inputs[0].dtype),weight=str(module.weight.dtype),output=str(output.dtype))
            if role in ('student','fake'):
                assert module.weight.dtype==torch.float32,(key,value)
                if not amp:assert inputs[0].dtype==output.dtype==torch.float32,(key,value)
            elif role=='teacher':assert module.weight.dtype==torch.bfloat16,(key,value)
            if key not in precision_seen:
                precision_seen[key]=value
                write(out/f'precision_rank{rank}.json',precision_seen)
        return hook
    for role,net in [('student',model.feedforward_model),('fake',model.guidance_model.fake_unet),('teacher',model.guidance_model.real_unet)]:
        net.conv_in.register_forward_hook(precision_hook(role))
    @torch.no_grad()
    def evaluate(update):
        from PIL import Image
        cases=[('train',i) for i in orders['student'][:4]]+[
            ('holdout',data['original_indices'].index(i)) for i in manifest['eval_original_indices'][:4]]
        dest=out/'eval'/f'g{update:06d}';dest.mkdir(parents=True,exist_ok=True);rows=[]
        with torch.random.fork_rng(devices=[device.index]):
            for split,idx in cases[rank::world]:
                emb=data['prompt_embeds'][idx:idx+1].to(device,dtype=torch.float32)
                pooled=data['pooled_prompt_embeds'][idx:idx+1].to(device,dtype=torch.float32)
                torch.manual_seed(102)
                x=torch.randn(1,4,128,128,device=device)
                knots=[999,749,499,249,-1]
                for t,s in zip(knots[:-1],knots[1:]):
                    tt=torch.tensor([t],device=device)
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        eps=model.feedforward_model(x,tt,emb,added_cond_kwargs={'text_embeds':pooled,'time_ids':model.add_time_ids}).sample.float()
                    x0=schedule.clean(x,eps,tt)
                    x=schedule.noise(x0,torch.randn_like(x0),s) if s>=0 else x0
                assert torch.isfinite(x).all()
                rgb=model.vae.decode(x/model.vae.config.scaling_factor).sample
                assert torch.isfinite(rgb).all()
                filename=f'{split}_{data["original_indices"][idx]}.png'
                Image.fromarray(((rgb[0]/2+.5).clamp(0,1).permute(1,2,0)*255).round().byte().cpu().numpy()).save(dest/filename)
                rows.append(dict(split=split,original_index=data['original_indices'][idx],seed=102,image=filename,prompt=data['prompts'][idx]))
        write(dest/f'rank{rank}.json',dict(G=update,samples=rows,sampler='official DMD2 x0 prediction + independent Gaussian renoise; four-step; BF16 autocast evaluation'))
        dist.barrier()
    if rank==0:
        write(out/'config.json',dict(cfg,micro=a.micro,accumulation=accum,world=world,target_G=a.updates,
              target_F=a.updates*5,smoke=a.smoke,method='official DMD2 clean DM + native shared fake GAN head',
              inference_schedule=[999,749,499,249],training_rollout='native stochastic backward simulation',
              actual_fake_noise_range=[0,999],student_DM_range=[20,979],PCM=0,mix=0))
    evaluate(0)
    model.train()  # Official Trainer.train_one_step enters train mode.
    torch.cuda.reset_peak_memory_stats();start=time.monotonic()
    for iteration in range(a.updates*5):
        tick=time.monotonic();do_g=(iteration%5==0)
        factor=min(iteration*cfg['warmup_reference_processes']/cfg['warmup_scheduler_steps'],1.)
        for group in og.param_groups:group['lr']=cfg['student_lr']*factor
        for group in of.param_groups:group['lr']=cfg['fake_lr']*factor
        selected=iteration%4 if a.smoke else random.Random(seed_for(seed,'depth',iteration,0)).randrange(4)
        model.selected_training_step=selected
        # One official training iteration produces G samples, optionally updates G,
        # then updates fake+head on their detached versions (upstream order).
        og.zero_grad(set_to_none=True);of.zero_grad(set_to_none=True)
        cache=[];gl=[0.,0.];fl=[0.,0.]
        for m in range(accum):
            role='student' if do_g else 'fake';b=batch(role,g if do_g else f,m);real=batch('real',f,m)
            torch.manual_seed(seed_for(seed,'forward',iteration*accum+m,rank))
            noise=torch.randn(a.micro,4,128,128,device=device)
            losses,logs=model(noise,None,None,visual=False,denoising_dict=b,real_train_dict=real,
                compute_generator_gradient=do_g,generator_turn=True,guidance_turn=False)
            model.guidance_model.real_unet.requires_grad_(False)
            if do_g:
                # Keep fake frozen during activation recomputation as in its G forward.
                model.guidance_model.requires_grad_(False)
            cache.append(logs['guidance_data_dict'])
            if do_g:
                total=losses['loss_dm']+cfg['gen_gan_weight']*losses['gen_cls_loss']
                if not torch.isfinite(total):raise FloatingPointError('Non-finite student loss')
                (total/accum).backward()
                model.guidance_model.fake_unet.requires_grad_(True)
                model.guidance_model.cls_pred_branch.requires_grad_(True)
                gl[0]+=losses['loss_dm'].detach().item()/accum;gl[1]+=losses['gen_cls_loss'].detach().item()/accum
                del total
            del losses,logs,b,real,noise
        if do_g:
            gn=clip(student_params,cfg['grad_clip']);og.step();g+=1
            og.zero_grad(set_to_none=True);of.zero_grad(set_to_none=True)
        for item in cache:
            losses,logs=model(None,None,None,generator_turn=False,guidance_turn=True,guidance_data_dict=item)
            total=losses['loss_fake_mean']+cfg['fake_gan_weight']*losses['guidance_cls_loss']
            if not torch.isfinite(total):raise FloatingPointError('Non-finite fake loss')
            (total/accum).backward()
            fl[0]+=losses['loss_fake_mean'].detach().item()/accum;fl[1]+=losses['guidance_cls_loss'].detach().item()/accum
            del losses,logs,total
        fn=clip(fake_params,cfg['grad_clip']);of.step();of.zero_grad(set_to_none=True);f+=1
        del cache,item
        torch.cuda.synchronize();seconds=time.monotonic()-tick
        vals=metrics_reduce(gl+fl)
        record(dict(iteration=iteration,G=g,F=f,student_updated=do_g,depth=selected,
                    student_lr=og.param_groups[0]['lr'],fake_lr=of.param_groups[0]['lr'],
                    student_dm=vals[0],student_gan=vals[1],fake_mse=vals[2],fake_gan=vals[3],
                    student_grad=gn if do_g else None,fake_grad=fn,seconds=seconds,
                    peak_GiB=torch.cuda.max_memory_allocated()/2**30))
        if rank==0:
            write(out/'status.json',dict(status='running',G=g,F=f,target_G=a.updates,target_F=a.updates*5,
                  seconds=time.monotonic()-start,last_iteration_seconds=seconds,peak_GiB=torch.cuda.max_memory_allocated()/2**30))
            print('UPDATE',iteration,'G',g,'F',f,'sec',round(seconds,2),flush=True)
        if f%5==0 and g%2==0:evaluate(g)
        stop=torch.tensor(int((out/'STOP_REQUEST').exists()),device=device);dist.all_reduce(stop,op=dist.ReduceOp.MAX)
        if stop.item():raise RuntimeError('Stop requested; experiment incomplete')
    assert g==a.updates and f==a.updates*5
    # Training is finished and only inference weights are requested. Release
    # Adam moments before gathering the student for export (no math changes).
    del og,of
    torch.cuda.empty_cache()
    state=get_model_state_dict(model.feedforward_model,options=StateDictOptions(full_state_dict=True,cpu_offload=True))
    if rank==0:
        ck=out/'student.pt';torch.save({k:v.to(torch.bfloat16) for k,v in state.items()},ck);del state
        weights=torch.load(ck,map_location='cpu',weights_only=True,mmap=True)
        with torch.device('meta'):check=UNet2DConditionModel.from_config(json.loads((Path(cfg['model_id'])/'unet/config.json').read_text()))
        check.load_state_dict(weights,strict=True,assign=True)
        assert all(torch.isfinite(v).all() for v in weights.values())
        write(out/'complete.json',dict(G=g,F=f,checkpoint=str(ck),strict_load=True,finite=True,elapsed_seconds=time.monotonic()-start))
        write(out/'status.json',dict(status='complete',G=g,F=f))
    dist.barrier();log.close();dist.destroy_process_group()

if __name__=='__main__':main()
