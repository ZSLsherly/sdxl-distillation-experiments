"""Paired COCO10K smoke images and timing, not a converged quality benchmark."""
import gc,io,json,os,pickle,time,zipfile
from pathlib import Path
import torch
from diffusers import StableDiffusionXLPipeline,UNet2DConditionModel,DDIMScheduler,LCMScheduler
from safetensors.torch import load_file
from sdxl_objectives import VPSchedule
from sdxl_prepare import DataUnpickler

@torch.inference_mode()
def main():
    rank=int(os.environ.get('LOCAL_RANK',0));torch.cuda.set_device(rank);torch.set_num_threads(4)
    root=Path('/data5/shenglin.zeng/sdxl_crossdistill');base=root/'models/sdxl-base-1.0';out=root/'outputs/smoke_1024/comparison';out.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(root/'datasets/dmd2/coco10k.zip') as archive:
        member=next(n for n in archive.namelist() if n.endswith('all_prompts.pkl'))
        prompts=DataUnpickler(io.BytesIO(archive.read(member))).load()
    assert isinstance(prompts,list) and len(prompts)==10000
    prompt=prompts[rank];assert isinstance(prompt,str)
    train=torch.load(root/'datasets/prepared.pt',map_location='cpu',weights_only=True)
    assert prompt not in set(train['prompts'])|set(train['pcm_prompts']);del train
    pipe=StableDiffusionXLPipeline.from_pretrained(base,variant='fp16',torch_dtype=torch.bfloat16,local_files_only=True).to('cuda')
    pipe.vae.to(torch.float32);pipe.set_progress_bar_config(disable=True)
    pe,ne,pp,np=pipe.encode_prompt(prompt,device='cuda',do_classifier_free_guidance=True)
    encoded={'prompt_embeds':pe,'negative_prompt_embeds':ne,'pooled_prompt_embeds':pp,'negative_pooled_prompt_embeds':np}
    conditions={'text_embeds':pp,'time_ids':torch.tensor([[1024,1024,0,0,1024,1024]],device='cuda',dtype=torch.bfloat16)}
    original_scheduler=pipe.scheduler
    schedule=VPSchedule(DDIMScheduler.from_pretrained(base,subfolder='scheduler',local_files_only=True).alphas_cumprod)
    records=[]
    methods=[('teacher_50step',None),('dmd2_4step',root/'models/baselines/dmd2/unet.safetensors'),('senseflow_4step',root/'models/baselines/senseflow/unet.safetensors'),('crossdistill_short_4step',root/'outputs/smoke_1024/export/unet/diffusion_pytorch_model.safetensors')]
    for name,checkpoint in methods:
        if checkpoint:
            state=load_file(checkpoint,device='cpu');pipe.unet.load_state_dict(state,strict=True);del state;gc.collect()
        pipe.scheduler=original_scheduler if name=='teacher_50step' else LCMScheduler.from_config(original_scheduler.config)
        for seed in [9,10,11]:
            generator=torch.Generator(device='cuda').manual_seed(seed)
            noise=torch.randn((1,4,128,128),generator=generator,device='cuda',dtype=torch.float32).to(torch.bfloat16)
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();tick=time.perf_counter();denoise_seconds=None
            if name=='crossdistill_short_4step':
                x=noise.float()
                for t,s in zip([999,749,499,249],[749,499,249,-1]):
                    tt=torch.tensor([t],device='cuda',dtype=torch.long)
                    eps=pipe.unet(x.to(torch.bfloat16),tt,encoder_hidden_states=pe,added_cond_kwargs=conditions,return_dict=False)[0].float()
                    x=schedule.jump(x,eps,tt,s)
                torch.cuda.synchronize();denoise_seconds=time.perf_counter()-tick
            else:
                end=[]
                def callback(p,i,t,k):
                    if i==(49 if name=='teacher_50step' else 3):
                        torch.cuda.synchronize();end.append(time.perf_counter()-tick)
                    return k
                kwargs={'num_inference_steps':50,'guidance_scale':8.0} if name=='teacher_50step' else {'num_inference_steps':4,'guidance_scale':0.0,'timesteps':[999,749,499,249]}
                x=pipe(**encoded,height=1024,width=1024,latents=noise,generator=generator,callback_on_step_end=callback,output_type='latent',**kwargs).images
                denoise_seconds=end[-1]
            # Decode every method through the same FP32 VAE. The stock pipeline
            # only auto-upcasts FP16, and does not cast BF16 latents to FP32.
            image=pipe.vae.decode(x.float()/pipe.vae.config.scaling_factor,return_dict=False)[0]
            images=pipe.image_processor.postprocess(image,output_type='pil')
            torch.cuda.synchronize();seconds=time.perf_counter()-tick
            assert len(images)==1
            if seed!=9:
                path=out/f'{name}_prompt{rank}_seed{seed}.png';images[0].save(path)
                row={'method':name,'prompt_index':rank,'prompt':prompt,'seed':seed,'resolution':1024,'denoise_seconds':denoise_seconds,'end_to_end_seconds':seconds,'peak_allocated_GiB':torch.cuda.max_memory_allocated()/1024**3,'image':str(path),'warmup_excluded':True,'NFE':100 if name=='teacher_50step' else 4,'status':'short-trained CrossDistill is not converged' if name.startswith('crossdistill') else 'official pretrained checkpoint'}
                records.append(row);print(json.dumps(row),flush=True)
    (out/f'timing_rank{rank}.json').write_text(json.dumps({'rank':rank,'source':'DMD2 official COCO2014 10K evaluation package','source_member':member,'total_official_prompts':len(prompts),'evaluated_prompts_per_rank':1,'seeds':[10,11],'precision':'BF16 UNet and CLIP; FP32 VAE','records':records},indent=2))
    print('PAIRED_COMPARISON_COMPLETE',rank,flush=True)

if __name__=='__main__':main()
