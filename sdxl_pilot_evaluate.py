"""Held-out pilot assessment; never interpret 16 samples as a benchmark FID."""
import gc,json,time,statistics
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from diffusers import StableDiffusionXLPipeline,LCMScheduler
from transformers import CLIPTokenizer
from safetensors.torch import load_file
from torchvision.transforms import Compose,Resize,CenterCrop,ToTensor,Normalize,InterpolationMode
from openai_clip_model import build_model

ROOT=Path('/data5/shenglin.zeng/sdxl_crossdistill');OUT=ROOT/'outputs/hparam_pilot'
TRIALS=['lr5e7_pcm10','lr25e8_pcm10','lr5e7_pcm1','lr25e8_pcm1']

@torch.inference_mode()
def main():
    torch.cuda.set_device(0);torch.set_num_threads(4)
    base=ROOT/'models/sdxl-base-1.0';dest=OUT/'assessment';dest.mkdir(exist_ok=True)
    pipe=StableDiffusionXLPipeline.from_pretrained(base,variant='fp16',torch_dtype=torch.bfloat16,local_files_only=True).to('cuda')
    pipe.vae.to(torch.float32);pipe.set_progress_bar_config(disable=True)
    original_scheduler=pipe.scheduler
    data=torch.load(ROOT/'datasets/prepared.pt',weights_only=True,map_location='cpu')
    cases=[];latents={}
    for rank in range(4):
        before=torch.load(OUT/TRIALS[0]/f'validation_before_rank{rank}.pt',weights_only=True)
        refs=torch.load(OUT/'validation_refs'/f'tau749_rank{rank}.pt',weights_only=True)
        for row,ref in zip(before,refs):
            key=(row['cache_index'],row['seed']);cases.append({'key':key,'prompt':row['prompt']})
            latents.setdefault('base_4step',{})[key]=row['final']
            latents.setdefault('teacher_ddim',{})[key]=ref['final']
        for trial in TRIALS:
            after=torch.load(OUT/trial/f'validation_after_rank{rank}.pt',weights_only=True)
            for row in after:latents.setdefault(trial,{})[(row['cache_index'],row['seed'])]=row['final']
    def decode(x):
        value=pipe.vae.decode(x.to('cuda',dtype=torch.float32)/pipe.vae.config.scaling_factor,return_dict=False)[0]
        return pipe.image_processor.postprocess(value,output_type='pil')[0]
    records=[]
    for method,mapping in latents.items():
        for case in cases:
            idx,seed=case['key'];path=dest/f'{method}_idx{idx}_seed{seed}.png'
            decode(mapping[(idx,seed)]).save(path)
            records.append({'method':method,'cache_index':idx,'seed':seed,'prompt':case['prompt'],'image':str(path)})
        print('DECODED',method,len(cases),flush=True)
    # Official pretrained baselines use their own documented stochastic LCM sampler.
    for method in ['dmd2','senseflow']:
        state=load_file(ROOT/f'models/baselines/{method}/unet.safetensors');pipe.unet.load_state_dict(state,strict=True);del state;gc.collect()
        pipe.scheduler=LCMScheduler.from_config(original_scheduler.config)
        for case in cases:
            idx,seed=case['key'];generator=torch.Generator(device='cuda').manual_seed(seed)
            noise=torch.randn((1,4,128,128),device='cuda',dtype=torch.float32,generator=generator).to(torch.bfloat16)
            encoded={k:data[k][idx:idx+1].to('cuda') for k in ['prompt_embeds','negative_prompt_embeds','pooled_prompt_embeds','negative_pooled_prompt_embeds']}
            torch.cuda.synchronize();start=time.perf_counter()
            x=pipe(**encoded,height=1024,width=1024,latents=noise,generator=generator,num_inference_steps=4,timesteps=[999,749,499,249],guidance_scale=0,output_type='latent').images
            torch.cuda.synchronize();seconds=time.perf_counter()-start
            path=dest/f'{method}_idx{idx}_seed{seed}.png';decode(x).save(path)
            records.append({'method':method,'cache_index':idx,'seed':seed,'prompt':case['prompt'],'image':str(path),'denoise_seconds':seconds})
        print('BASELINE_COMPLETE',method,flush=True)
    del pipe,data,latents;gc.collect();torch.cuda.empty_cache()
    model=build_model(torch.jit.load(str(ROOT/'models/metrics/ViT-B-32.pt'),map_location='cpu').state_dict()).eval().to('cuda')
    tokenizer=CLIPTokenizer.from_pretrained(base,subfolder='tokenizer',local_files_only=True)
    transform=Compose([Resize(224,interpolation=InterpolationMode.BICUBIC),CenterCrop(224),ToTensor(),Normalize((.48145466,.4578275,.40821073),(.26862954,.26130258,.27577711))])
    features={};text_features={}
    for row in records:
        prompt=row['prompt']
        if prompt not in text_features:
            ids=tokenizer.encode(prompt,add_special_tokens=True,truncation=True,max_length=77)
            tokens=torch.zeros((1,77),device='cuda',dtype=torch.long);tokens[0,:len(ids)]=torch.tensor(ids,device='cuda')
            f=model.encode_text(tokens).float();text_features[prompt]=f/f.norm(dim=-1,keepdim=True)
        image_input=transform(Image.open(row['image']).convert('RGB')).unsqueeze(0).to('cuda')
        f=model.encode_image(image_input).float();f=f/f.norm(dim=-1,keepdim=True)
        row['clip_cosine']=(f*text_features[prompt]).sum().item()
        features[(row['method'],row['cache_index'],row['seed'])]=f.cpu()
    methods=list(dict.fromkeys(r['method'] for r in records));metrics={}
    for method in methods:
        rows=[r for r in records if r['method']==method]
        diversity=[]
        for idx in sorted(set(r['cache_index'] for r in rows)):
            diversity.append(1-(features[(method,idx,101)]*features[(method,idx,102)]).sum().item())
        metrics[method]={'clip_cosine_mean':statistics.mean(r['clip_cosine'] for r in rows),'seed_clip_cosine_distance_mean':statistics.mean(diversity),'images':len(rows),'prompts':len(diversity)}
        if method in TRIALS:
            summary=json.loads((OUT/method/'summary.json').read_text());logs=[];validation=[]
            for rank in range(4):
                logs.extend(json.loads(line) for line in (OUT/method/f'rank{rank}.jsonl').read_text().splitlines())
                validation.extend(json.loads((OUT/method/f'validation_after_rank{rank}.json').read_text()))
            student=[r for r in logs if r['phase']=='student'];fake=[r for r in logs if r['phase']=='fake_score']
            metrics[method].update({'training':summary,'boundary_relative_mse_mean':statistics.mean(r['boundary_relative_mse'] for r in validation),
                'student_step_seconds_median':statistics.median(r['seconds'] for r in student),'fake_step_seconds_median':statistics.median(r['seconds'] for r in fake),
                'student_clipped_fraction':statistics.mean(r['clip_scale']<1 for r in student),'fake_clipped_fraction':statistics.mean(r['clip_scale']<1 for r in fake),
                'dmd_loss_mean':statistics.mean(r['dmd_loss'] for r in student),'pcm_loss_mean':statistics.mean(r['pcm_loss'] for r in student),
                'validation_finite':all(r['finite'] for r in validation)})
            # Resample prompts, keeping both seeds paired, to avoid pseudo-replication.
            differences=[]
            for idx in sorted(set(r['cache_index'] for r in rows)):
                own=statistics.mean(r['clip_cosine'] for r in rows if r['cache_index']==idx)
                initial=statistics.mean(r['clip_cosine'] for r in records if r['method']=='base_4step' and r['cache_index']==idx)
                differences.append(own-initial)
            rng=np.random.default_rng(20261017);boot=rng.choice(differences,size=(10000,len(differences)),replace=True).mean(1)
            metrics[method]['clip_gain_vs_untrained']=float(np.mean(differences));metrics[method]['clip_gain_95pct_prompt_bootstrap_ci']=np.quantile(boot,[.025,.975]).tolist()
    report={'data':'DMD2 LAION prompt cache indices 96..103, disjoint from training indices 0..95','prompts':8,'seeds':[101,102],'images_per_method':16,
        'resolution':1024,'clip_model':'OpenAI ViT-B/32, SHA256 verified','clip_tokenizer':'SDXL CLIP-L vocab/merges, zero padding, causal text encoder, first EOT pooling',
        'metrics':metrics,'records':records,'caveats':['Pilot sample size cannot support FID or a converged quality ranking.',
        'Teacher-relative final error is not an optimization target for DMD.','Crossover 749 is provisional pending human review; four candidate diagnostics are supplied.',
        'Official pretrained DMD2 and SenseFlow have vastly larger training budgets; these are visual references.','Teacher diagnostic uses DDIM for VP-coordinate consistency; earlier smoke teacher used Euler.',
        'No GAN loss: CrossDistill PCM-high + DMD-low instantiation; DMD2 supplies optimizer and critic settings, not its entire method.']}
    (dest/'result.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
    print('PILOT_ASSESSMENT_COMPLETE',json.dumps({k:{n:v for n,v in m.items() if n!='training'} for k,m in metrics.items()}),flush=True)

if __name__=='__main__':main()
