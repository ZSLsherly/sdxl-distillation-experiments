"""Expand the original SDXL 10K split without changing cached old samples or holdout."""
import hashlib,json,os,time,sys
from pathlib import Path
import numpy as np
import torch
from torch.nn.functional import cosine_similarity
from diffusers import StableDiffusionXLPipeline
from sdxl_prepare import load_numpy_pt
from batching import conditions

root=Path(os.environ['SDXL_ROOT']);work=Path(os.environ['SDXL_WORK'])
old_dir=root/'datasets/parallel_10k_seed20261018'
name='parallel_20k_seed20261018'
datasets=work/'datasets';out=datasets/name;workdir=datasets/(name+'.building')
cache=workdir/'parts'
workdir.mkdir(exist_ok=True);cache.mkdir(exist_ok=True)
def read(p):return json.loads(p.read_text())
def save_pt(p,x):
    tmp=p.with_suffix('.partial');torch.save(x,tmp);tmp.replace(p)
def load(p):return torch.load(p,map_location='cpu',weights_only=True,mmap=True)
def canon(s):return ' '.join(s.lower().split())
def digest(x):return hashlib.sha256(x).hexdigest()
def log(*a):print(*a,flush=True)
def delta(old,meta,reference):
    if (cache/'selected.pt').exists(): return load(cache/'selected.pt')
    assert meta['source_shard_offsets']==[0,8000,16000]
    seen_p={canon(p) for p in old['prompts']}
    seen_l={digest(x.numpy().tobytes()) for x in old['latents']}
    assert len(seen_l)==len(seen_p)==10032
    choices=np.random.default_rng(meta['seed']).permutation(16000)
    files=[work/'datasets/20k_sources'/f'vae_latents_{i:03d}.pt' for i in (2,3)]
    shards=[load_numpy_pt(f) for f in files]
    assert all(len(s['prompts'])==len(s['latents'])==8000 and s['latents'].shape[1:]==(4,128,128) and s['latents'].dtype==np.float16 for s in shards)
    prompts=[];ids=[];keys=[];latents=torch.empty((10000,4,128,128),dtype=old['latents'].dtype)
    for val in choices:
        val=int(val);part,row=divmod(val,8000);s=shards[part]
        sentence=str(s['prompts'][row]);key=canon(sentence)
        if not key or key in seen_p or key in reference: continue
        z=s['latents'][row];h=digest(z.tobytes())
        if h in seen_l or not np.isfinite(z).all():continue
        seen_l.add(h);seen_p.add(key)
        latents[len(ids)].copy_(torch.from_numpy(z))
        prompts.append(sentence);ids.append(16000+val);keys.append(f'{files[part].name}:{row}')
        if len(ids)==10000:break
    assert len(ids)==10000
    result=dict(prompts=prompts,original_indices=ids,source_keys=keys,latents=latents)
    save_pt(cache/'selected.pt',result);log('SELECTED',len(ids));return result

@torch.inference_mode()
def encode(pipe,prompts):
    pe,_,pp,_=pipe.encode_prompt(prompts,device='cpu',do_classifier_free_guidance=False)
    return pe.bfloat16(),pp.bfloat16()

@torch.inference_mode()
def encode_delta(old,delta):
    pipe=StableDiffusionXLPipeline.from_pretrained(work/'bc_texture_inputs/models/sdxl-base-1.0',
        unet=None,vae=None,variant='fp16',torch_dtype=torch.bfloat16,local_files_only=True)
    indexes=[0,311,920,3456,7890,9999,10000,10030]
    report={};good=[]
    for mode,dtype in [('bf16',torch.bfloat16),('fp32_from_bf16',torch.float32)]:
        pipe.text_encoder.to(dtype);pipe.text_encoder_2.to(dtype)
        b=time.monotonic();x,y=encode(pipe,[old['prompts'][i] for i in indexes]);elapsed=time.monotonic()-b
        values=[]
        for new,key in [(x,'prompt_embeds'),(y,'pooled_prompt_embeds')]:
            a=old[key][indexes].float().flatten(1);b_=new.float().flatten(1)
            values.append(float(cosine_similarity(a,b_).min()))
        report[mode]=dict(seconds=elapsed,min_token_embedding_cos=values[0],min_pooled_cos=values[1])
        if min(values)>.995:good.append((elapsed,mode,dtype))
    log('ENCODER_CHECK',report);assert good,'Encoder cache differs materially from original'
    chosen_file=cache/'encoding.json'
    if chosen_file.exists(): mode=read(chosen_file)['chosen'];dtype={'bf16':torch.bfloat16,'fp32_from_bf16':torch.float32}[mode];assert any(m==mode for _,m,_ in good)
    else:
        _,mode,dtype=min(good)
        chosen_file.write_text(json.dumps({'chosen':mode,'test':report,'old_reference_indices':indexes},indent=2))
    pipe.text_encoder.to(dtype);pipe.text_encoder_2.to(dtype)
    log('ENCODING',mode)
    for start in range(0,10000,128):
        path=cache/f'enc_{start:05}.pt'
        if path.exists(): continue
        embeds=[];pooled=[];stop=min(10000,start+128)
        for i in range(start,stop,16):
            x,y=encode(pipe,delta['prompts'][i:min(i+16,stop)]);embeds.append(x);pooled.append(y)
        x,y=torch.cat(embeds),torch.cat(pooled)
        assert x.shape==(stop-start,77,2048) and y.shape==(stop-start,1280)
        assert torch.isfinite(x).all() and torch.isfinite(y).all()
        save_pt(path,dict(x=x,y=y));log('ENCODED',stop,10000)
def build():
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    if out.exists():
        assert (out/'validation.json').is_file();log('ALREADY_READY',out);return
    meta=read(old_dir/'manifest.json');old=load(old_dir/'prepared.pt')
    assert old['train_count']==10000 and len(old['prompts'])==10032
    coco=read(root/'outputs/bc_winner_late_mix_20261102/coco10k/manifest.json')
    ref={canon(r['prompt']) for r in coco['samples']}
    d=delta(old,meta,ref);encode_delta(old,d)
    data=dict(old);data['train_count']=20000
    for key in ('original_indices','source_keys','prompts'):
        data[key]=old[key][:10000]+list(d[key])+old[key][10000:]
    for key in ('latents','prompt_embeds','pooled_prompt_embeds'):
        src=old[key];dest=torch.empty((20032,*src.shape[1:]),dtype=src.dtype)
        dest[:10000].copy_(src[:10000]);dest[20000:].copy_(src[10000:])
        if key=='latents':dest[10000:20000].copy_(d[key])
        else:
            for start in range(0,10000,128):
                chunk=load(cache/f'enc_{start:05}.pt')['x' if key=='prompt_embeds' else 'y']
                dest[10000+start:10000+start+len(chunk)].copy_(chunk)
        data[key]=dest
    save_pt(workdir/'prepared.pt',data);del data,d,dest,src
    v=load(workdir/'prepared.pt')
    for key in old:
        if key in ('latents','prompt_embeds','pooled_prompt_embeds','original_indices','source_keys','prompts'):
            if torch.is_tensor(old[key]):
                assert torch.equal(v[key][:10000],old[key][:10000]) and torch.equal(v[key][20000:],old[key][10000:])
                assert all(torch.isfinite(z).all() for z in v[key].split(64))
            else:assert v[key][:10000]==old[key][:10000] and v[key][20000:]==old[key][10000:]
        elif key!='train_count':assert torch.equal(v[key],old[key])
    texts=[canon(s) for s in v['prompts']]; hash_l={digest(z.numpy().tobytes()) for z in v['latents']}
    assert len(set(texts))==len(hash_l)==len(set(v['original_indices']))==20032
    assert not set(texts[10000:20000])&ref
    assert v['original_indices'][20000:]==meta['holdout_original_indices']
    cond=conditions(v,[0,9999,10000,19999,20000,20031],'cpu')
    assert cond['prompt_embeds'].shape==(6,77,2048) and cond['time_ids'].shape==(6,6)
    manifest=dict(meta,source='DMD2 LAION latent shards 000–003; first 10K and holdout from existing 10K cache, additions from 002/003',
        source_samples=32000,train_count=20000,train_original_indices=v['original_indices'][:20000],
        source_shard_offsets=[0,8000,16000,24000,32000],source_files=[f'vae_latents_{i:03d}.pt' for i in range(4)],
        parent_dataset=str(old_dir),added_sources_sha256={'002':'dd26be16f0d5c8067fdefae8909434cb65008232d4463b9cb4be509318df51ba','003':'e7ce92fdc7d3a7be2379e5bad082ff70265d7f23836ab4b22f36b0ce29deb6ad'},
        added_exact_COCO_prompts_excluded=True,encoding=read(cache/'encoding.json'))
    (workdir/'manifest.json').write_text(json.dumps(manifest,indent=2))
    val=dict(train=20000,holdout=32,unique_prompts=20032,unique_latents=len(hash_l),old_samples_bit_identical=True,
        exact_training_caption_COCO_overlap=len(set(texts[:20000])&ref),new_exact_COCO_overlap=0,loader_check='passed',
        bytes=(workdir/'prepared.pt').stat().st_size)
    (workdir/'validation.json').write_text(json.dumps(val,indent=2))
    # Delete precisely this build's regenerable intermediate files; retain the pinned input shards.
    for p in [cache/'selected.pt',cache/'encoding.json',*(cache/f'enc_{i:05}.pt' for i in range(0,10000,128))]:p.unlink()
    cache.rmdir();workdir.rename(out);log('READY',str(out),val)
if __name__=='__main__':build()
