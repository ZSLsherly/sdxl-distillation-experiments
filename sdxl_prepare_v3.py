"""Encode the existing full real-LAION shard; reserve exact-deduplicated holdout."""
import json,hashlib,os
from pathlib import Path
import numpy as np
import torch
from diffusers import StableDiffusionXLPipeline
from sdxl_prepare import load_numpy_pt

root=Path('/data5/shenglin.zeng/sdxl_crossdistill');out=root/'datasets/gan_v3';out.mkdir(exist_ok=True)
if (out/'manifest.json').exists():
    print('V3_DATA_ALREADY_PREPARED',flush=True);raise SystemExit
torch.set_num_threads(4)
data=load_numpy_pt(root/'datasets/dmd2/vae_latents_000.pt')
old=torch.load(root/'datasets/prepared.pt',map_location='cpu',weights_only=True)
canon=lambda s:' '.join(str(s).lower().split())
hold=list(dict.fromkeys(old['original_indices'][96:128]))
hold_prompts={canon(data['prompts'][i]) for i in hold}
hold_hashes={hashlib.sha256(data['latents'][i].tobytes()).hexdigest() for i in hold}
seen_p=set(hold_prompts);seen_h=set(hold_hashes);train=[]
for i in np.random.default_rng(20261017).permutation(len(data['latents'])):
    p=canon(data['prompts'][i]);h=hashlib.sha256(data['latents'][i].tobytes()).hexdigest()
    if p in seen_p or h in seen_h:continue
    train.append(int(i));seen_p.add(p);seen_h.add(h)
order=train+hold;prompts=[str(data['prompts'][i]) for i in order]
pipe=StableDiffusionXLPipeline.from_pretrained(root/'models/sdxl-base-1.0',unet=None,vae=None,variant='fp16',torch_dtype=torch.bfloat16,local_files_only=True)
pipe.text_encoder.to('cuda');pipe.text_encoder_2.to('cuda')
result={'latents':torch.from_numpy(data['latents'][order].copy()),'prompts':prompts,'original_indices':order,'train_count':len(train),'time_ids':torch.tensor([[1024,1024,0,0,1024,1024]],dtype=torch.float32)}
positive=[];pooled=[]
with torch.inference_mode():
    for start in range(0,len(order),16):
        pe,ne,pp,npool=pipe.encode_prompt(prompts[start:start+16],device='cuda',do_classifier_free_guidance=True)
        positive.append(pe.cpu());pooled.append(pp.cpu())
        if start==0:result.update(negative_prompt_embeds=ne[:1].cpu(),negative_pooled_prompt_embeds=npool[:1].cpu())
        if start%256==0:print('ENCODE',start,len(order),flush=True)
result.update(prompt_embeds=torch.cat(positive),pooled_prompt_embeds=torch.cat(pooled))
torch.save(result,out/'prepared.pt.tmp');(out/'prepared.pt.tmp').replace(out/'prepared.pt')
(out/'manifest.json').write_text(json.dumps({'source':'existing official DMD2 LAION latent shard 000','source_samples':len(data['latents']),'train_count':len(train),'holdout_count':len(hold),'train_original_indices':train,'holdout_original_indices':hold,'eval_original_indices':[old['original_indices'][i] for i in [113,118,114,119]],'train_eval_exact_prompt_and_latent_overlap':0,'dedup':'canonical exact captions and SHA256 of latent bytes; no perceptual duplicate detection','initialization':'fresh SDXL Base; old pilot weights not used'},indent=2))
print('V3_DATA_READY',len(train),len(hold),flush=True)
