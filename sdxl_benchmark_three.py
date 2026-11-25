"""Extend the original fixed-prompt benchmark; CPU only, cached by image hash."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import csv,json,hashlib,time
from pathlib import Path
from datetime import datetime
import sdxl_benchmark_curves as base
from sdxl_benchmark_curves import torch,np,Image,build_model,CLIPTokenizer,Compose,Resize,CenterCrop,ToTensor,Normalize,InterpolationMode
ROOT=base.ROOT
OLD=ROOT/'outputs/benchmark_curves_20261018'
OUT=ROOT/'outputs/benchmark_curves_three_20261018'
OUT.mkdir(exist_ok=True)

@torch.inference_mode()
def main():
    original=json.loads((OLD/'manifest.json').read_text())
    commit=(ROOT/'src/benchmark_vendor/source_commit.txt').read_text()
    assert original['basicsr_commit']==commit
    launches=[json.loads((ROOT/'outputs'/p/'launch.json').read_text()) for p in ['gan_stage_parallel','gan_stage_joint34']]
    inventory=[];runs={};identity=None
    for launch in launches:
        for run in launch['runs']:
            p=ROOT/'outputs'/run['run'];config=json.loads((p/'config.json').read_text())
            runs[run['variant']]={**run,'launch_timestamp':launch['timestamp'],'world_size':config['world_size'],'config':config}
            for d in sorted((p/'eval').glob('*')):
                files=[d/f'rank{r}.json' for r in range(config['world_size'])]
                if not all(f.exists() for f in files):continue
                docs=[json.loads(f.read_text()) for f in files]
                rows=sorted([r for doc in docs for r in doc['samples'] if r['split']=='holdout'],key=lambda r:r['original_index'])
                if len(rows)!=4 or not all((d/r['image']).is_file() for r in rows):continue
                ids=[(r['original_index'],r['prompt'],r['seed']) for r in rows]
                if identity is None:identity=ids
                assert ids==identity,(run,d,'Evaluation cases differ')
                g=docs[0]['student_updates'];assert all(doc['student_updates']==g for doc in docs)
                inventory.append(dict(variant=run['variant'],directory=str(d),student_updates=g,student_samples=docs[0]['student_samples'],timestamp=d.name.split('_g')[0],rows=rows))
    manifest=dict(created_at=datetime.now().astimezone().isoformat(),runs=runs,inventory=inventory,basicsr_commit=commit,niqe_sanity=original['niqe_sanity'],metrics=original['metrics'],scope=original['scope'],cache_source=str(OLD/'per_image.jsonl'))
    cache={}
    for src in [OLD/'per_image.jsonl',OUT/'per_image.jsonl']:
        if src.exists():
            for line in src.read_text().splitlines():
                r=json.loads(line);cache[(r['image_sha256'],r['original_index'],r['prompt'])]={m:r[m] for m in ['niqe','clip_cosine']}
    (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    print('INVENTORY',json.dumps({v:dict(points=sum(r['variant']==v for r in inventory),last_g=max(r['student_updates'] for r in inventory if r['variant']==v)) for v in runs}),'CACHED_IMAGES',len(cache),flush=True)
    model=build_model(torch.jit.load(str(ROOT/'models/metrics/ViT-B-32.pt'),map_location='cpu').state_dict()).float().eval()
    tokenizer=CLIPTokenizer.from_pretrained(ROOT/'models/sdxl-base-1.0/tokenizer',local_files_only=True)
    transform=Compose([Resize(224,interpolation=InterpolationMode.BICUBIC),CenterCrop(224),ToTensor(),Normalize((.48145466,.4578275,.40821073),(.26862954,.26130258,.27577711))])
    texts={}
    for idx,prompt,seed in identity:
        ids=tokenizer.encode(prompt,add_special_tokens=True,truncation=True,max_length=77)
        tokens=torch.zeros((1,77),dtype=torch.long);tokens[0,:len(ids)]=torch.tensor(ids)
        f=model.encode_text(tokens).float();texts[idx]=(f/f.norm(dim=-1,keepdim=True))[0]
    start=time.time();all_rows=[];computed=0;hits=0
    with (OUT/'per_image.jsonl').open('w') as out:
        for i,rec in enumerate(inventory):
            paths=[Path(rec['directory'])/r['image'] for r in rec['rows']]
            images=[Image.open(p).convert('RGB') for p in paths]
            keys=[(hashlib.sha256(p.read_bytes()).hexdigest(),r['original_index'],r['prompt']) for p,r in zip(paths,rec['rows'])]
            needed=[j for j,k in enumerate(keys) if k not in cache];hits+=len(keys)-len(needed)
            if needed:
                # Fixed four-image CLIP batch matches the original scoring setup.
                f=model.encode_image(torch.stack([transform(im) for im in images])).float();f=f/f.norm(dim=-1,keepdim=True)
                for j in needed:
                    cache[keys[j]]=dict(niqe=base.score_niqe(images[j]),clip_cosine=float((f[j]*texts[rec['rows'][j]['original_index']]).sum()))
                    computed+=1
            for row,p,im,key in zip(rec['rows'],paths,images,keys):
                result={k:rec[k] for k in ['variant','student_updates','student_samples','timestamp']}
                result.update(original_index=row['original_index'],seed=row['seed'],prompt=row['prompt'],image=str(p),width=im.width,height=im.height,image_sha256=key[0],**cache[key])
                assert np.isfinite(result['niqe']) and np.isfinite(result['clip_cosine'])
                all_rows.append(result);out.write(json.dumps(result,ensure_ascii=False)+'\n')
            out.flush()
            if i%10==0 or i==len(inventory)-1:print('PROGRESS',i+1,len(inventory),rec['variant'],rec['student_updates'],'new_images',computed,'cached',hits,'seconds',round(time.time()-start),flush=True)
    summary=[]
    for rec in inventory:
        rs=[r for r in all_rows if r['variant']==rec['variant'] and r['student_updates']==rec['student_updates']]
        item={k:rec[k] for k in ['variant','student_updates','student_samples','timestamp']};item['n']=len(rs)
        for metric in ['niqe','clip_cosine']:
            vals=[r[metric] for r in rs];item[metric]=float(np.mean(vals));item[metric+'_sd']=float(np.std(vals,ddof=1))
        summary.append(item)
    for name,rows in [('per_image',all_rows),('summary',summary)]:
        with (OUT/f'{name}.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    print('COMPLETE',len(summary),len(all_rows),'new_images',computed,'cached',hits,'seconds',round(time.time()-start),flush=True)
if __name__=='__main__':main()
