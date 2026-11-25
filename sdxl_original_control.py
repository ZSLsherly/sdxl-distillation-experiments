"""Inference-only untouched SDXL controls, alongside saved pilot outputs."""
import argparse,json,time
from pathlib import Path
import torch
from diffusers import StableDiffusionXLPipeline,DDIMScheduler
from sdxl_objectives import VPSchedule

@torch.inference_mode()
def main():
    a=argparse.ArgumentParser();a.add_argument('--idx',type=int,required=True);args=a.parse_args()
    torch.set_num_threads(4);torch.cuda.set_device(0)
    root=Path('/data5/shenglin.zeng/sdxl_crossdistill');base=root/'models/sdxl-base-1.0'
    out=root/'outputs/original_sdxl_control';out.mkdir(exist_ok=True)
    data=torch.load(root/'datasets/prepared.pt',weights_only=True,map_location='cpu');idx=args.idx;seed=102
    pipe=StableDiffusionXLPipeline.from_pretrained(base,variant='fp16',torch_dtype=torch.bfloat16,local_files_only=True).to('cuda')
    pipe.vae.to(torch.float32);pipe.set_progress_bar_config(disable=True)
    c={k:data['pcm_'+k][idx:idx+1].to('cuda') for k in ['prompt_embeds','negative_prompt_embeds','pooled_prompt_embeds','negative_pooled_prompt_embeds']}
    noise=torch.randn((1,4,128,128),device='cuda',generator=torch.Generator(device='cuda').manual_seed(seed))
    def save(x,name):
        rgb=pipe.vae.decode(x.to('cuda',dtype=torch.float32)/pipe.vae.config.scaling_factor,return_dict=False)[0]
        pipe.image_processor.postprocess(rgb,output_type='pil')[0].save(out/f'{idx}_{name}.png')
    scheduler_config=dict(pipe.scheduler.config)
    x=pipe(**c,height=1024,width=1024,latents=noise.to(torch.bfloat16).clone(),num_inference_steps=50,guidance_scale=8,output_type='latent').images
    save(x,'base_50_cfg8');print(idx,'BASE_50_DONE',flush=True)
    sched=VPSchedule(DDIMScheduler.from_pretrained(base,subfolder='scheduler',local_files_only=True).alphas_cumprod)
    def predict(x,t,negative=False):
        with torch.autocast('cuda',dtype=torch.bfloat16):
            return pipe.unet(x,t,encoder_hidden_states=c['negative_prompt_embeds' if negative else 'prompt_embeds'],added_cond_kwargs={'text_embeds':c['negative_pooled_prompt_embeds' if negative else 'pooled_prompt_embeds'],'time_ids':data['time_ids'][idx:idx+1].to('cuda')},return_dict=False)[0].float()
    for cfg in [1,8]:
        x=noise.clone()
        for t,s in zip([999,919,499,199],[919,499,199,-1]):
            tt=torch.tensor([t],device='cuda');ec=predict(x,tt)
            if cfg!=1:
                eu=predict(x,tt,True);ec=eu+cfg*(ec-eu)
            x=sched.jump(x,ec,tt,s)
        save(x,f'base_4_cfg{cfg}');print(idx,'BASE_4',cfg,'DONE',flush=True)
    series=root/'outputs/gan_stage_pilot_v2'
    for name,trial,when in [('before_v2','dmd','before'),('after_A','dmd','after'),('after_B','gan4','after'),('after_C','joint4','after'),('after_D','joint34_matched','after')]:
        rows=torch.load(series/trial/f'validation_{when}_rank{(idx-112)%4}.pt',weights_only=True,map_location='cpu')
        row=next(r for r in rows if r['cache_index']==idx and r['seed']==seed)
        assert row['prompt']==data['pcm_prompts'][idx]
        save(row['final'],name)
    (out/f'{idx}_metadata.json').write_text(json.dumps({'idx':idx,'seed':seed,'prompt':data['pcm_prompts'][idx],'base_source':str(base),'base_50_scheduler':type(pipe.scheduler).__name__,'base_50_scheduler_config':scheduler_config,'four_step_knots':[999,919,499,199,-1],'four_step_cfg':1,'before_v2':'12 prior student updates; not untouched SDXL','after':'24 additional updates per variant','note':'Native 50-step pipeline is a normal-generation quality reference; raw base 4-step CFG1 uses the identical pilot sampler, float32 initial noise and conditional UNet calls. CFG8 is an extra guidance control. All decoded with the same full-frame FP32 VAE.'},indent=2))
    print('CONTROL_COMPLETE',idx,flush=True)
if __name__=='__main__':main()
