"""Existing evaluator, six disjoint global block shards; fixed COCO1024 CLIP."""
from pathlib import Path
from types import SimpleNamespace
import json,os,sys,time
import numpy as np
import torch
from transformers import CLIPConfig,CLIPModel,CLIPTokenizer,CLIPImageProcessor
import evaluation_core as core

ROOT=Path(sys.argv[1]);ARM='A_clean_global_high30_v2';OUT=ROOT/'results'/ARM
REV='3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268'
def read(p):return json.loads(Path(p).read_text())
def unit(t):return torch.nn.functional.normalize(t.float(),dim=-1).cpu().numpy()
def clip_model():
    md=ROOT/'models/clip';assert (md/'revision.txt').read_text().strip()==REV
    model=CLIPModel(CLIPConfig.from_pretrained(md,local_files_only=True)).eval()
    weights=torch.load(md/'pytorch_model.bin',map_location='cpu',weights_only=True)
    for k in ('vision_model.embeddings.position_ids','text_model.embeddings.position_ids'):
        if k not in model.state_dict():weights.pop(k,None)
    model.load_state_dict(weights,strict=True)
    return model,CLIPTokenizer.from_pretrained(md,local_files_only=True),CLIPImageProcessor.from_pretrained(md,local_files_only=True)
def setup():
    manifest=read(OUT/'manifest.json');sel=read(ROOT/'selection.json')
    assert manifest['n']==10000 and sel['revision']==REV and len(sel['cases'])==1024
    assert len({x['index'] for x in sel['cases']})==1024
    for x in sel['cases']:
        y=manifest['samples'][x['index']]
        assert x==dict(index=y['index'],seed=y['seed'],prompt=y['prompt'])
    core.ROOT=ROOT;core.BASE=ROOT/'models/sdxl-base-1.0';core.OUT=OUT
    core.RUNS=manifest['models'];core.METHODS=[ARM]
    return manifest,sel['cases']
def generate_rank(rank,world):
    from resource_guard import guard,ResourceConflict
    uuid=os.environ['CUDA_VISIBLE_DEVICES'];assert uuid==read(ROOT/'node.json')['gpus'][rank]
    guard(uuid);manifest,cases=setup();chosen={x['index'] for x in cases}
    receipt=read(ROOT/'assets_ready.json');weight=ROOT/'checkpoint/student_rank0.pt'
    assert (weight.stat().st_size,weight.stat().st_mtime_ns)==(receipt['bytes'],receipt['mtime_ns'])
    model,_,proc=clip_model();clipdir=OUT/'clip_features';clipdir.mkdir(exist_ok=True)
    original=core.generate
    @torch.inference_mode()
    def generate(pipe,method,prompts,seeds):
        guard(uuid);images,info=original(pipe,method,prompts,seeds)
        picked=[(j,s-core.SEED) for j,s in enumerate(seeds) if s-core.SEED in chosen and not (clipdir/f'{s-core.SEED:05d}.npy').exists()]
        if picked:
            x=proc(images=[images[j] for j,i in picked],return_tensors='pt')['pixel_values']
            features=unit(model.visual_projection(model.vision_model(pixel_values=x).pooler_output))
            assert np.isfinite(features).all()
            for (j,i),v in zip(picked,features):
                p=clipdir/f'{i:05d}.npy'
                with p.with_suffix('.partial').open('wb') as f:np.save(f,v)
                p.with_suffix('.partial').replace(p)
        return images,info
    core.generate=generate
    try:core.worker(SimpleNamespace(rank=rank,world=world,limit=0,probe=False,methods=ARM,batch=2,teacher_batch=2))
    except ResourceConflict as e:print('RESOURCE_CONFLICT',e,flush=True);raise SystemExit(75)
def finalize():
    manifest,cases=setup();model,tok,_=clip_model()
    iv=np.stack([np.load(OUT/'clip_features'/f"{r['index']:05d}.npy") for r in cases])
    assert iv.shape==(1024,512) and np.isfinite(iv).all()
    tv=[]
    with torch.inference_mode():
        for i in range(0,len(cases),16):
            x=tok([r['prompt'] for r in cases[i:i+16]],padding=True,truncation=True,max_length=77,return_tensors='pt')
            tv.append(unit(model.text_projection(model.text_model(**x).pooler_output)))
    sim=iv@np.concatenate(tv).T;pos=sim.diagonal()*100;prompts=[r['prompt'] for r in cases]
    ranks=np.array([1+np.count_nonzero(sim[i]>max(sim[i,j] for j,t in enumerate(prompts) if t==p)) for i,p in enumerate(prompts)])
    core.write_json(OUT/'clip_results.json',dict(n=1024,revision=REV,cases=cases,
        metrics={ARM:dict(cosine_x100=float(pos.mean()),top1_correct=int((ranks==1).sum()),top5_correct=int((ranks<=5).sum()))},
        per_case={ARM:dict(alignment=pos.tolist(),rank=ranks.tolist())}))
    core.metrics();scores=read(OUT/'scores.json')[ARM]
    assert scores['n']==10000 and np.isfinite([scores['FID'],scores['Patch_FID'],float(pos.mean())]).all()
    core.write_json(OUT/'all_complete.json',dict(arm=ARM,**scores,CLIP=float(pos.mean()),clip_n=1024,finished_unix=time.time()))
    print('ALL_METRICS_COMPLETE',scores,float(pos.mean()),flush=True)
if __name__=='__main__':
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    if sys.argv[2]=='finalize':finalize()
    else:
        from resource_guard import ResourceConflict
        try:generate_rank(int(sys.argv[2]),int(sys.argv[3]))
        except ResourceConflict as e:print('RESOURCE_CONFLICT',e,flush=True);raise SystemExit(75)
