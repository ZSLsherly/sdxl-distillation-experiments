import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,json,csv,hashlib,time
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import Compose,Resize,CenterCrop,ToTensor,Normalize,InterpolationMode
ROOT=Path('/data5/shenglin.zeng/sdxl_crossdistill')
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'src/benchmark_vendor')]
from openai_clip_model import build_model
from transformers import CLIPTokenizer
from niqe_core import niqe
from color_util import rgb2ycbcr

OUT=ROOT/'outputs/benchmark_curves_20261018'
OUT.mkdir(exist_ok=True)
torch.set_num_threads(2)
params=np.load(ROOT/'src/benchmark_vendor/niqe_pris_params.npz')
def score_niqe(im):
    y=rgb2ycbcr(np.asarray(im).astype(np.float32)/255.,y_only=True)*255.
    return niqe(y.round(),params['mu_pris_param'],params['cov_pris_param'],params['gaussian_window'])

@torch.inference_mode()
def main():
    sanity=score_niqe(Image.open(ROOT/'src/benchmark_vendor/baboon.png').convert('RGB'))
    assert abs(sanity-5.7295763)<0.001, sanity
    print('NIQE_SANITY',sanity,flush=True)
    launch=json.loads((ROOT/'outputs/gan_stage_parallel/launch.json').read_text())
    inventory=[];identity=None
    for run in launch['runs']:
        p=ROOT/'outputs'/run['run']
        for d in sorted((p/'eval').glob('*')):
            files=sorted(d.glob('rank*.json'))
            if len(files)!=2:continue
            docs=[json.loads(f.read_text()) for f in files]
            rows=[r for doc in docs for r in doc['samples'] if r['split']=='holdout']
            if len(rows)!=4 or not all((d/r['image']).is_file() for r in rows):continue
            g=docs[0]['student_updates']
            # Fixed snapshot, identical to the inventory announced before evaluation.
            if g>({'joint4':114,'gan4':154}[run['variant']]):continue
            ids=sorted((r['original_index'],r['prompt'],r['seed']) for r in rows)
            if identity is None:identity=ids
            assert ids==identity,(run,d,'changed evaluation cases')
            assert all(doc['student_updates']==g for doc in docs)
            inventory.append(dict(variant=run['variant'],directory=str(d),student_updates=g,student_samples=docs[0]['student_samples'],timestamp=d.name.split('_g')[0],rows=rows))
    manifest=dict(created_at=datetime.now().astimezone().isoformat(),launch=launch,inventory=inventory,niqe_sanity=sanity,basicsr_commit=(ROOT/'src/benchmark_vendor/source_commit.txt').read_text(),scope='4 fixed held-out prompts, one fixed seed each; exploratory trend, not an official dataset benchmark',metrics={'niqe':'BasicSR NIQE Y channel, original resolution, no border crop; lower better','clip_cosine':'OpenAI CLIP ViT-B/32 FP32 cosine, official 224 center crop; higher better'})
    (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    print('INVENTORY',len(inventory),flush=True)
    model=build_model(torch.jit.load(str(ROOT/'models/metrics/ViT-B-32.pt'),map_location='cpu').state_dict()).float().eval()
    tokenizer=CLIPTokenizer.from_pretrained(ROOT/'models/sdxl-base-1.0/tokenizer',local_files_only=True)
    transform=Compose([Resize(224,interpolation=InterpolationMode.BICUBIC),CenterCrop(224),ToTensor(),Normalize((.48145466,.4578275,.40821073),(.26862954,.26130258,.27577711))])
    text_features={}
    for idx,prompt,seed in identity:
        ids=tokenizer.encode(prompt,add_special_tokens=True,truncation=True,max_length=77)
        tokens=torch.zeros((1,77),dtype=torch.long);tokens[0,:len(ids)]=torch.tensor(ids)
        f=model.encode_text(tokens).float();text_features[idx]=(f/f.norm(dim=-1,keepdim=True))[0]
    start=time.time();all_rows=[];cache={}
    # Share the exactly identical G0 images' metric computation across groups.
    with (OUT/'per_image.jsonl').open('w') as out:
        for i,rec in enumerate(inventory):
            paths=[Path(rec['directory'])/r['image'] for r in rec['rows']]
            images=[Image.open(p).convert('RGB') for p in paths]
            keys=[(hashlib.sha256(p.read_bytes()).hexdigest(),r['original_index']) for p,r in zip(paths,rec['rows'])]
            needed=[j for j,k in enumerate(keys) if k not in cache]
            if needed:
                f=model.encode_image(torch.stack([transform(images[j]) for j in needed])).float();f=f/f.norm(dim=-1,keepdim=True)
                for j,feat in zip(needed,f):
                    cache[keys[j]]=dict(niqe=score_niqe(images[j]),clip_cosine=float((feat*text_features[rec['rows'][j]['original_index']]).sum()))
            for row,p,im,k in zip(rec['rows'],paths,images,keys):
                result={k:rec[k] for k in ['variant','student_updates','student_samples','timestamp']}
                result.update(original_index=row['original_index'],seed=row['seed'],prompt=row['prompt'],image=str(p),width=im.width,height=im.height,image_sha256=k[0],**cache[k])
                assert np.isfinite(result['niqe']) and np.isfinite(result['clip_cosine'])
                all_rows.append(result);out.write(json.dumps(result,ensure_ascii=False)+'\n')
            out.flush()
            if i%5==0 or i==len(inventory)-1:print('PROGRESS',i+1,len(inventory),rec['variant'],rec['student_updates'],'seconds',round(time.time()-start),flush=True)
    with (OUT/'per_image.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(all_rows[0]));w.writeheader();w.writerows(all_rows)
    summary=[]
    for rec in inventory:
        rs=[r for r in all_rows if r['variant']==rec['variant'] and r['student_updates']==rec['student_updates']]
        item={k:rec[k] for k in ['variant','student_updates','student_samples','timestamp']};item['n']=len(rs)
        for metric in ['niqe','clip_cosine']:
            vals=[r[metric] for r in rs];item[metric]=float(np.mean(vals));item[metric+'_sd']=float(np.std(vals,ddof=1))
        summary.append(item)
    with (OUT/'summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    print('COMPLETE',len(summary),len(all_rows),round(time.time()-start),flush=True)
if __name__=='__main__':main()
