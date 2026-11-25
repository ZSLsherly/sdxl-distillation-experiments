"""Paired teacher diagnostics in native SDXL VP/DDIM coordinates."""
import gc,io,json,os,zipfile
from pathlib import Path
import torch
from diffusers import StableDiffusionXLPipeline,DDIMScheduler
from sdxl_objectives import VPSchedule
from sdxl_prepare import DataUnpickler

@torch.inference_mode()
def main():
    rank=int(os.environ.get('LOCAL_RANK',0));torch.cuda.set_device(rank);torch.set_num_threads(4)
    root=Path('/data5/shenglin.zeng/sdxl_crossdistill');base=root/'models/sdxl-base-1.0'
    out=root/'outputs/hparam_pilot/teacher_stages';out.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(root/'datasets/dmd2/coco10k.zip') as z:
        member=next(n for n in z.namelist() if n.endswith('all_prompts.pkl'))
        prompts=DataUnpickler(io.BytesIO(z.read(member))).load()
    prompt=prompts[rank]
    pipe=StableDiffusionXLPipeline.from_pretrained(base,variant='fp16',torch_dtype=torch.bfloat16,local_files_only=True).to('cuda')
    pipe.vae.to(torch.float32);pipe.set_progress_bar_config(disable=True)
    pe,ne,pp,np=pipe.encode_prompt(prompt,device='cuda',do_classifier_free_guidance=True)
    tids=torch.tensor([[1024,1024,0,0,1024,1024]],device='cuda',dtype=torch.bfloat16)
    del pipe.text_encoder,pipe.text_encoder_2;gc.collect();torch.cuda.empty_cache()
    schedule=VPSchedule(DDIMScheduler.from_pretrained(base,subfolder='scheduler',local_files_only=True).alphas_cumprod)
    candidates=[899,749,599,449]
    grid=sorted(set(range(999,-1,-10))|set(candidates)|{0},reverse=True)
    def predict(x,t):
        xx=torch.cat([x,x]).to(torch.bfloat16)
        eps=pipe.unet(xx,t,encoder_hidden_states=torch.cat([ne,pe]),added_cond_kwargs={'text_embeds':torch.cat([np,pp]),'time_ids':tids.repeat(2,1)},return_dict=False)[0].float()
        eu,ec=eps.chunk(2);return eu+8*(ec-eu)
    def decode(x,path):
        value=pipe.vae.decode(x.float()/pipe.vae.config.scaling_factor,return_dict=False)[0]
        pipe.image_processor.postprocess(value,output_type='pil')[0].save(path)
    records=[]
    for seed in [10,11,12,13]:
        generator=torch.Generator(device='cuda').manual_seed(seed)
        x=torch.randn((1,4,128,128),generator=generator,device='cuda',dtype=torch.float32)
        stored={};rows=[]
        for t,s in zip(grid,grid[1:]+[-1]):
            tt=torch.tensor([t],device='cuda',dtype=torch.long);eps=predict(x,tt)
            if t in candidates:
                clean=schedule.clean(x,eps,tt);stored[t]=clean.cpu()
                alpha=schedule.alpha[t].item();sigma=schedule.sigma[t].item()
                rows.append({'t':t,'alpha':alpha,'sigma':sigma,'log_snr':float(torch.log(schedule.alpha[t]**2/schedule.sigma[t]**2)),
                             'predicted_clean_image':str(out/f'prompt{rank}_seed{seed}_t{t}_x0.png'),
                             'noisy_state_image':str(out/f'prompt{rank}_seed{seed}_t{t}_xt.png')})
                decode(clean,rows[-1]['predicted_clean_image']);decode(x,rows[-1]['noisy_state_image'])
            x=schedule.jump(x,eps,tt,s)
        decode(x,out/f'prompt{rank}_seed{seed}_final.png')
        for row in rows:
            delta=stored[row['t']]-x.cpu()
            row['clean_estimate_vs_final_latent_rmse']=delta.square().mean().sqrt().item()
        torch.save({'predicted_clean':stored,'final':x.cpu()},out/f'prompt{rank}_seed{seed}_latents.pt')
        records.append({'prompt_index':rank,'prompt':prompt,'seed':seed,'stages':rows,'final_image':str(out/f'prompt{rank}_seed{seed}_final.png')})
        print('TEACHER_TRAJECTORY_DONE',rank,seed,flush=True)
    (out/f'rank{rank}.json').write_text(json.dumps({'records':records,'candidates':candidates,'solver':'deterministic DDIM eta=0 using native VP alpha/sigma',
        'teacher_cfg':8,'inference_grid':grid,'resolution':1024,'meaning':'x0 images are one-step clean estimates at the indicated state, not final images or raw noisy states',
        'selection':'visual diagnostic only; no automatically validated optimal crossover'},indent=2))
    print('TEACHER_STAGES_COMPLETE',rank,flush=True)

if __name__=='__main__':main()
