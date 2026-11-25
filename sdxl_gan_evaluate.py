"""Assess completed PixelGAN pilots on an independent paired LAION holdout.

No small-sample FID. High-frequency energy is a diagnostic, not a quality score.
Official pretrained baselines are references with different training budgets.
"""
import gc,json,statistics,time,hashlib
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from diffusers import AutoencoderKL,StableDiffusionXLPipeline,LCMScheduler
from transformers import CLIPTokenizer
from safetensors.torch import load_file
from torchvision.transforms import Compose,Resize,CenterCrop,ToTensor,Normalize,InterpolationMode
from openai_clip_model import build_model
from sdxl_pixel_gan import PixelPatchDiscriminator,decode_tile

ROOT=Path('/data5/shenglin.zeng/sdxl_crossdistill')
OUT=ROOT/'outputs/gan_stage_pilot';TRIALS=['dmd','gan4','joint4','joint34']

def paired_bootstrap(deltas):
    values=np.asarray(deltas);rng=np.random.default_rng(20261017)
    samples=rng.choice(values,size=(10000,len(values)),replace=True).mean(1)
    return {'mean':float(values.mean()),'prompt_bootstrap_95pct_ci':np.quantile(samples,[.025,.975]).tolist()}

@torch.inference_mode()
def main():
    torch.cuda.set_device(0);torch.set_num_threads(4)
    base=ROOT/'models/sdxl-base-1.0';dest=OUT/'assessment';dest.mkdir(exist_ok=True)
    data=torch.load(ROOT/'datasets/prepared.pt',weights_only=True,map_location='cpu')
    mapping={};cases=[];audit={'status':'checking','trials':{}}
    for rank in range(4):
        initial=torch.load(OUT/'dmd'/f'validation_before_rank{rank}.pt',weights_only=True)
        refs=torch.load(OUT/'validation_refs_112_119'/f'rank{rank}.pt',weights_only=True)
        for row,ref in zip(initial,refs):
            key=(row['cache_index'],row['seed']);assert 112<=key[0]<120
            assert (ref['cache_index'],ref['seed'])==key
            cases.append({'key':key,'prompt':row['prompt']})
            mapping.setdefault('initial_crossdistill',{})[key]=row
            mapping.setdefault('teacher_ddim',{})[key]={'final':ref['final'],'boundary':ref['boundary']}
        for trial in TRIALS:
            before=torch.load(OUT/trial/f'validation_before_rank{rank}.pt',weights_only=True)
            after=torch.load(OUT/trial/f'validation_after_rank{rank}.pt',weights_only=True)
            assert len(before)==len(initial)==len(after)==4
            for init,row,own in zip(initial,before,after):
                assert row['prompt']==init['prompt'] and row['cache_index']==init['cache_index'] and row['seed']==init['seed']
                assert torch.equal(row['final'],init['final']),'Initial outputs differ across variants'
                mapping.setdefault(trial,{})[(own['cache_index'],own['seed'])]=own
    assert len(cases)==16 and len({c['key'] for c in cases})==16
    for trial in TRIALS:
        summary=json.loads((OUT/trial/'summary.json').read_text())
        args=json.loads((OUT/trial/'args.json').read_text());split=json.loads((OUT/trial/'data_split.json').read_text())
        assert summary['student_updates']==24 and summary['fake_updates']==120 and summary['status']=='training_complete'
        assert summary['config']['schedule']==[999,919,499,199,-1]
        assert split['evaluated_validation_indices']==list(range(112,120))
        records=[]
        for rank in range(4):
            rows=[json.loads(line) for line in (OUT/trial/f'rank{rank}.jsonl').read_text().splitlines()]
            g=[r for r in rows if r['phase']=='student'];d=[r for r in rows if r['phase']!='student']
            assert len(g)==24 and len(d)==120 and [r['generator_stage'] for r in g]==[2,3,4]*8
            assert all(np.isfinite(r['loss']) and np.isfinite(r['grad_norm']) for r in rows)
            assert all(r['cache_index']<96 for r in rows)
            assert len({r['cache_index'] for r in g})==24
            assert all(r.get('gan_latent_gradient_rms',1)>0 for r in g if r['gan_active'])
            assert sum(r['gan_active'] for r in g)=={'dmd':0,'gan4':8,'joint4':8,'joint34':16}[trial]
            assert sum(r['dmd_active'] for r in g)==(16 if trial=='gan4' else 24)
            assert any(r.get('r1',0)>0 for r in d)
            rank_summary=json.loads((OUT/trial/f'summary_rank{rank}.json').read_text())
            assert rank_summary['student_parameter_delta']>0
            records.extend(rows)
        student=[r for r in records if r['phase']=='student'];critic=[r for r in records if r['phase']!='student']
        audit['trials'][trial]={'all_four_ranks_updated':True,'student_unique_train_images':len({r['cache_index'] for r in student}),'initial_outputs_identical':True,'source_sha256':json.loads((OUT/trial/'source_sha256.json').read_text()),'summary':summary,
            'student_step_seconds_median':statistics.median(r['seconds'] for r in student),'fake_plus_D_step_seconds_median':statistics.median(r['seconds'] for r in critic),
            'student_clip_fraction':statistics.mean(r['clip_scale']<1 for r in student),'fake_clip_fraction':statistics.mean(r['clip_scale']<1 for r in critic),
            'gan_gradient_rms_mean':statistics.mean(r['gan_latent_gradient_rms'] for r in student if r['gan_active']) if trial!='dmd' else None,
            'gan_to_dmd_latent_gradient_ratio_median':statistics.median(summary['config']['gan_weight']*r['gan_latent_gradient_rms']/r['dmd_latent_gradient_rms'] for r in student if r['gan_active'] and r['dmd_active']) if trial in ['joint4','joint34'] else None}
    assert len({json.dumps(audit['trials'][t]['source_sha256'],sort_keys=True) for t in TRIALS})==1
    audit['status']='passed';(OUT/'audit.json').write_text(json.dumps(audit,indent=2))
    print('AUDIT_PASSED',flush=True)
    vae=AutoencoderKL.from_pretrained(base,subfolder='vae',variant='fp16',torch_dtype=torch.float32,local_files_only=True).eval().to('cuda')
    def decode(x):
        value=vae.decode(x.to('cuda',dtype=torch.float32)/vae.config.scaling_factor,return_dict=False)[0]
        value=(value/2+.5).clamp(0,1).cpu()[0]
        return Image.fromarray((value.permute(1,2,0).numpy()*255).round().astype(np.uint8))
    image_records=[]
    for method,latents in mapping.items():
        for case in cases:
            idx,seed=case['key'];row=latents[(idx,seed)];path=dest/f'{method}_idx{idx}_seed{seed}.png'
            decode(row['final']).save(path)
            image_records.append({'method':method,'cache_index':idx,'seed':seed,'prompt':case['prompt'],'image':str(path)})
            if method in TRIALS and idx in [112,114,116,118] and seed==101:
                for stage,x in enumerate(row['clean_estimates'],1):decode(x).save(dest/f'{method}_idx{idx}_seed{seed}_stage{stage}.png')
        print('DECODED',method,flush=True)
    discriminator_diagnostics={}
    for trial in TRIALS:
        disc=PixelPatchDiscriminator().to('cuda').eval().requires_grad_(False)
        disc.load_state_dict(torch.load(OUT/trial/'discriminator.pt',weights_only=True,map_location='cpu'))
        scores={'train_real':[],'heldout_real':[],'heldout_fake':[]}
        for idx in list(range(8))+list(range(112,120)):
            real=data['latents'][idx:idx+1].to('cuda',dtype=torch.float32)
            condition=data['pcm_pooled_prompt_embeds'][idx:idx+1].to('cuda')
            logit=disc(decode_tile(vae,real,(40,40)),condition).item()
            scores['train_real' if idx<96 else 'heldout_real'].append(logit)
            if idx>=112:
                latent=mapping[trial][(idx,101)]['final'].to('cuda',dtype=torch.float32)
                scores['heldout_fake'].append(disc(decode_tile(vae,latent,(40,40)),condition).item())
        discriminator_diagnostics[trial]={k:statistics.mean(v) for k,v in scores.items()}
        discriminator_diagnostics[trial]['raw_logits']=scores
        del disc
    del vae;gc.collect();torch.cuda.empty_cache()
    pipe=StableDiffusionXLPipeline.from_pretrained(base,variant='fp16',torch_dtype=torch.bfloat16,local_files_only=True).to('cuda')
    pipe.vae.to(torch.float32);pipe.set_progress_bar_config(disable=True);original_scheduler=pipe.scheduler
    for method in ['dmd2','senseflow']:
        state=load_file(ROOT/f'models/baselines/{method}/unet.safetensors');pipe.unet.load_state_dict(state,strict=True);del state;gc.collect()
        pipe.scheduler=LCMScheduler.from_config(original_scheduler.config)
        for case in cases:
            idx,seed=case['key'];generator=torch.Generator(device='cuda').manual_seed(seed)
            noise=torch.randn((1,4,128,128),device='cuda',dtype=torch.float32,generator=generator).to(torch.bfloat16)
            c={k:data['pcm_'+k][idx:idx+1].to('cuda') for k in ['prompt_embeds','negative_prompt_embeds','pooled_prompt_embeds','negative_pooled_prompt_embeds']}
            x=pipe(**c,height=1024,width=1024,latents=noise,generator=generator,num_inference_steps=4,timesteps=[999,749,499,249],guidance_scale=0,output_type='latent').images
            rgb=pipe.vae.decode(x.to(torch.float32)/pipe.vae.config.scaling_factor,return_dict=False)[0]
            path=dest/f'{method}_idx{idx}_seed{seed}.png';pipe.image_processor.postprocess(rgb,output_type='pil')[0].save(path)
            image_records.append({'method':method,'cache_index':idx,'seed':seed,'prompt':case['prompt'],'image':str(path)})
        print('OFFICIAL_REFERENCE',method,flush=True)
    del pipe,data,mapping;gc.collect();torch.cuda.empty_cache()
    model=build_model(torch.jit.load(str(ROOT/'models/metrics/ViT-B-32.pt'),map_location='cpu').state_dict()).eval().to('cuda')
    tokenizer=CLIPTokenizer.from_pretrained(base,subfolder='tokenizer',local_files_only=True)
    transform=Compose([Resize(224,interpolation=InterpolationMode.BICUBIC),CenterCrop(224),ToTensor(),Normalize((.48145466,.4578275,.40821073),(.26862954,.26130258,.27577711))])
    texts={};features={};lowpasses={}
    for row in image_records:
        prompt=row['prompt'];key=(row['method'],row['cache_index'],row['seed'])
        if prompt not in texts:
            ids=tokenizer.encode(prompt,add_special_tokens=True,truncation=True,max_length=77)
            tokens=torch.zeros((1,77),device='cuda',dtype=torch.long);tokens[0,:len(ids)]=torch.tensor(ids,device='cuda')
            z=model.encode_text(tokens).float();texts[prompt]=F.normalize(z,dim=-1)
        img=Image.open(row['image']).convert('RGB');z=F.normalize(model.encode_image(transform(img).unsqueeze(0).to('cuda')).float(),dim=-1)
        row['clip_cosine']=(z*texts[prompt]).sum().item();features[key]=z.cpu()
        arr=np.asarray(img,dtype=np.float32)/255;gray=arr.mean(-1)
        lap=-4*gray[1:-1,1:-1]+gray[:-2,1:-1]+gray[2:,1:-1]+gray[1:-1,:-2]+gray[1:-1,2:]
        row['laplacian_variance']=float(lap.var());row['pixel_saturation_fraction']=float(((arr<.01)|(arr>.99)).mean())
        blur=np.asarray(img.resize((32,32),Image.Resampling.BOX),dtype=np.float32)/255
        lowpasses[key]=blur
    methods=list(dict.fromkeys(r['method'] for r in image_records));metrics={}
    by_key={(r['method'],r['cache_index'],r['seed']):r for r in image_records}
    for method in methods:
        rows=[r for r in image_records if r['method']==method]
        for row in rows:
            idx,seed=row['cache_index'],row['seed']
            row['lowpass_teacher_pair_rmse']=float(np.mean((lowpasses[(method,idx,seed)]-lowpasses[('teacher_ddim',idx,seed)])**2)**.5)
        diversity=[1-(features[(method,idx,101)]*features[(method,idx,102)]).sum().item() for idx in range(112,120)]
        metrics[method]={k:statistics.mean(r[k] for r in rows) for k in ['clip_cosine','laplacian_variance','pixel_saturation_fraction','lowpass_teacher_pair_rmse']}
        metrics[method]['seed_clip_cosine_distance']=statistics.mean(diversity)
        if method in TRIALS:
            val=[]
            for rank in range(4):val+=json.loads((OUT/method/f'validation_after_rank{rank}.json').read_text())
            metrics[method]['boundary_relative_mse']=statistics.mean(r['boundary_relative_mse'] for r in val)
            metrics[method]['training']=audit['trials'][method]
            for reference in ['initial_crossdistill','dmd']:
                changes=[statistics.mean(by_key[(method,idx,s)]['clip_cosine']-by_key[(reference,idx,s)]['clip_cosine'] for s in [101,102]) for idx in range(112,120)]
                metrics[method][f'clip_delta_vs_{reference}']=paired_bootstrap(changes)
    result={'protocol':'independent LAION paired caption holdout, cache indices 112..119, seeds 101/102','images_per_method':16,'training_resolution':1024,'metrics':metrics,'records':image_records,'discriminator_train_holdout_diagnostics':discriminator_diagnostics,
        'caveats':['PatchGAN from scratch, not SAM/DINO PixelGAN.','Real discriminator inputs are VAE reconstructions.','Tiled VAE decoding during training is approximate; evaluation decodes complete frames.','Only 24 G updates, 8 updates at each low-noise node; no convergence claim.','joint34 has twice the GAN exposure count at the same weight.','Pure GAN4 may need a separate weight sweep.','Laplacian variance can rise because of artifacts; not a perceptual quality score.','Teacher pair RMSE does not penalize all valid alternate generations.','Official DMD2 and SenseFlow reference models use their own LCM schedule and far larger training budgets.','No FID or PatchFID estimated from 16 samples.']}
    (dest/'result.json').write_text(json.dumps(result,indent=2,ensure_ascii=False))
    print('GAN_ASSESSMENT_COMPLETE',flush=True)

if __name__=='__main__':main()
