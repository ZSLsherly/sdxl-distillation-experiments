"""Preserve original 10K + holdout; add 40K unique official LAION samples."""
import argparse, hashlib, json, os, time, subprocess
from pathlib import Path
import numpy as np
import torch
from torch.nn.functional import cosine_similarity
from diffusers import StableDiffusionXLPipeline
from sdxl_prepare import load_numpy_pt

def canon(text): return ' '.join(str(text).lower().split())
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024**2),b''):h.update(chunk)
    return h.hexdigest()
def load(path):return torch.load(path,weights_only=True,map_location='cpu',mmap=True)
def save(path,value):
    tmp=path.with_suffix('.partial');torch.save(value,tmp);tmp.replace(path)
def log(*items):print(*items,flush=True)

def select(old,meta,source_dir,parts,reference):
    path=parts/'selected.pt'
    if path.exists():return load(path)
    specs=[s for s in json.loads((source_dir/'sources.json').read_text())['files']
           if (source_dir/s['name']).is_file()]
    specs.sort(key=lambda s:s['shard'])
    shards=[]
    for spec in specs:
        p=source_dir/spec['name']
        assert p.stat().st_size==spec['size'] and sha(p)==spec['sha256'],str(p)
        data=load_numpy_pt(p)
        assert len(data['prompts'])==8000 and data['latents'].shape==(8000,4,128,128)
        assert data['latents'].dtype==np.float16
        shards.append(data)
        log('LOADED_VERIFIED_SOURCE',spec['name'])
    seen_text={canon(s) for s in old['prompts']}
    seen_z={hashlib.sha256(z.numpy().tobytes()).hexdigest() for z in old['latents']}
    assert len(seen_text)==len(seen_z)==10032
    latents=torch.empty((40000,4,128,128),dtype=old['latents'].dtype)
    prompts=[];ids=[];keys=[];excluded=dict(duplicate_text=0,duplicate_latent=0,coco=0,invalid=0)
    for value in np.random.default_rng(meta['seed']).permutation(8000*len(shards)):
        part,row=divmod(int(value),8000);s=shards[part];spec=specs[part]
        prompt=str(s['prompts'][row]);key=canon(prompt)
        if not key or not np.isfinite(s['latents'][row]).all():excluded['invalid']+=1;continue
        if key in seen_text:excluded['duplicate_text']+=1;continue
        if key in reference:excluded['coco']+=1;continue
        z=s['latents'][row];digest=hashlib.sha256(z.tobytes()).hexdigest()
        if digest in seen_z:excluded['duplicate_latent']+=1;continue
        seen_text.add(key);seen_z.add(digest)
        latents[len(ids)].copy_(torch.from_numpy(z))
        prompts.append(prompt);ids.append(spec['shard']*8000+row);keys.append(f"{spec['name']}:{row}")
        if len(ids)==40000:break
    assert len(ids)==40000,f'Only {len(ids)} new unique samples; fetch fallback shard008'
    result=dict(latents=latents,prompts=prompts,original_indices=ids,source_keys=keys)
    save(path,result)
    (parts/'selection.json').write_text(json.dumps(dict(added=40000,excluded=excluded,
        source_files=[s['name'] for s in specs],source_hashes={s['name']:s['sha256'] for s in specs}),indent=2))
    log('SELECTED',40000,excluded)
    return result

@torch.inference_mode()
def encode(model,old,delta,parts,device):
    if device.startswith('cuda'):
        physical=os.environ['CUDA_VISIBLE_DEVICES']
        assert physical.isdigit(),'Use one explicitly selected physical GPU'
        uuid,used,util=subprocess.check_output(['nvidia-smi','-i',physical,
            '--query-gpu=uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(',')
        apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
        assert not any(line.split(',')[0].strip()==uuid.strip() for line in apps.splitlines()),'Selected GPU has another job'
        assert int(used)<256 and int(util)<=5,'Selected GPU is not idle'
        log('IDLE_GPU_CONFIRMED',physical,used,util)
    pipe=StableDiffusionXLPipeline.from_pretrained(model,unet=None,vae=None,
        variant='fp16',torch_dtype=torch.bfloat16,local_files_only=True)
    pipe.text_encoder.to(device);pipe.text_encoder_2.to(device)
    def call(prompts):
        pe,_,pp,_=pipe.encode_prompt(prompts,device=device,do_classifier_free_guidance=False)
        return pe.to('cpu',dtype=torch.bfloat16),pp.to('cpu',dtype=torch.bfloat16)
    indices=[0,311,920,3456,7890,9999,10000,10030]
    x,y=call([old['prompts'][i] for i in indices])
    checks=[float(cosine_similarity(new.float().flatten(1),old[key][indices].float().flatten(1)).min())
            for new,key in [(x,'prompt_embeds'),(y,'pooled_prompt_embeds')]]
    assert min(checks)>.995,checks
    (parts/'encoding.json').write_text(json.dumps(dict(device=device,dtype='bfloat16',
        min_token_cos=checks[0],min_pooled_cos=checks[1],old_reference_indices=indices),indent=2))
    log('ENCODER_CHECK',checks)
    for start in range(0,40000,256):
        path=parts/f'enc_{start:05d}.pt'
        if path.exists():continue
        stop=min(40000,start+256);xs=[];ys=[]
        for i in range(start,stop,32):
            x,y=call(delta['prompts'][i:min(i+32,stop)]);xs.append(x);ys.append(y)
        x,y=torch.cat(xs),torch.cat(ys)
        assert x.shape==(stop-start,77,2048) and y.shape==(stop-start,1280)
        assert torch.isfinite(x).all() and torch.isfinite(y).all()
        save(path,dict(x=x,y=y))
        log('ENCODED',stop,40000)
    del pipe
    if device.startswith('cuda'):torch.cuda.empty_cache()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--old-dir',type=Path,required=True)
    ap.add_argument('--source-dir',type=Path,required=True)
    ap.add_argument('--model',required=True)
    ap.add_argument('--coco-manifest',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--device',default='cuda')
    args=ap.parse_args();torch.set_num_threads(4);torch.set_num_interop_threads(1)
    out=args.output;build=out.with_name(out.name+'.building');parts=build/'parts'
    if out.exists():
        assert (out/'validation.json').is_file()
        validation=json.loads((out/'validation.json').read_text())
        assert validation['train']==50000 and validation['sha256']==sha(out/'prepared.pt')
        log('ALREADY_READY',out);return
    parts.mkdir(parents=True,exist_ok=True)
    old=load(args.old_dir/'prepared.pt');meta=json.loads((args.old_dir/'manifest.json').read_text())
    assert old['train_count']==10000 and len(old['prompts'])==10032
    coco=json.loads(args.coco_manifest.read_text())
    reference={canon(r['prompt']) for r in coco['samples']}
    assert len(coco['samples'])==10000
    assert not {canon(s) for s in old['prompts'][:10000]}&reference
    delta=select(old,meta,args.source_dir,parts,reference)
    encode(args.model,old,delta,parts,args.device)
    data=dict(old);data['train_count']=50000
    for key in ['original_indices','source_keys','prompts']:
        data[key]=old[key][:10000]+list(delta[key])+old[key][10000:]
    for key in ['latents','prompt_embeds','pooled_prompt_embeds']:
        src=old[key];dest=torch.empty((50032,*src.shape[1:]),dtype=src.dtype)
        dest[:10000].copy_(src[:10000]);dest[50000:].copy_(src[10000:])
        if key=='latents':dest[10000:50000].copy_(delta['latents'])
        else:
            for start in range(0,40000,256):
                chunk=load(parts/f'enc_{start:05d}.pt')['x' if key=='prompt_embeds' else 'y']
                dest[10000+start:10000+start+len(chunk)].copy_(chunk)
        data[key]=dest
    save(build/'prepared.pt',data);del data,delta,dest,src
    v=load(build/'prepared.pt')
    for key in old:
        if key in ['latents','prompt_embeds','pooled_prompt_embeds','original_indices','source_keys','prompts']:
            if torch.is_tensor(old[key]):
                assert torch.equal(v[key][:10000],old[key][:10000])
                assert torch.equal(v[key][50000:],old[key][10000:])
                assert all(torch.isfinite(z).all() for z in v[key].split(64))
            else:
                assert v[key][:10000]==old[key][:10000] and v[key][50000:]==old[key][10000:]
        elif key!='train_count':assert torch.equal(v[key],old[key])
    texts=[canon(s) for s in v['prompts']]
    unique_z={hashlib.sha256(z.numpy().tobytes()).hexdigest() for z in v['latents']}
    assert len(set(texts))==len(unique_z)==len(set(v['original_indices']))==50032
    assert not set(texts[:50000])&reference
    assert v['original_indices'][50000:]==meta['holdout_original_indices']
    for key in ['negative_prompt_embeds','negative_pooled_prompt_embeds','time_ids']:
        assert v[key].shape[0]==1 and torch.isfinite(v[key]).all()
    manifest=dict(meta,train_count=50000,holdout_count=32,
        source='Official DMD2 LAION SDXL scaled image latents with corresponding captions; existing 10K preserved, 40K added',
        parent_dataset=str(args.old_dir),parent_sha256=sha(args.old_dir/'prepared.pt'),
        coco_manifest_sha256=sha(args.coco_manifest),
        train_original_indices=v['original_indices'][:50000],
        selection=json.loads((parts/'selection.json').read_text()),
        encoding=json.loads((parts/'encoding.json').read_text()))
    # Update inherited source inventory so it cannot incorrectly describe only shards000/001.
    used=[int(n.split('_')[-1].split('.')[0]) for n in manifest['selection']['source_files']]
    manifest.update(source_files=[f'vae_latents_{i:03d}.pt' for i in [0,1]+used],
        source_samples=(2+len(used))*8000,source_shard_offsets=[8000*i for i in range(3+len(used))])
    validation=dict(train=50000,holdout=32,unique_prompts=len(set(texts)),unique_latents=len(unique_z),
        old_samples_bit_identical=True,exact_training_caption_COCO_overlap=0,
        bytes=(build/'prepared.pt').stat().st_size,sha256=sha(build/'prepared.pt'))
    (build/'manifest.json').write_text(json.dumps(manifest,indent=2))
    (build/'validation.json').write_text(json.dumps(validation,indent=2))
    (build/'数据集说明.txt').write_text('LAION衍生50K训练集＋原32条留出；非COCO评测集，非原始RGB下载。\n'
        '原10K与留出张量逐项一致；新增40K按标准化提示词及latent SHA256去重，排除固定COCO10K提示词完全重叠。\n'
        '不声明排除了近重复图片或语义近似；latent沿用官方VAE缩放，不重复缩放。\n'+json.dumps(validation,indent=2),encoding='utf-8')
    # Only remove this build's regenerable chunks after final data passed validation.
    for p in [parts/'selected.pt',*parts.glob('enc_*.pt')]:p.unlink()
    build.rename(out);log('READY',out,validation)
if __name__=='__main__':main()
