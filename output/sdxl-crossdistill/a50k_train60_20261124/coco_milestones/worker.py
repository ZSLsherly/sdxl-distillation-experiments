"""Existing COCO10K FID/Patch FID path, with CLIP on the fixed 1024 generated samples."""
import os,sys,json,time,subprocess,gc
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from transformers import CLIPConfig,CLIPModel,CLIPTokenizer,CLIPImageProcessor
import evaluation_core as core
ROOT=Path(sys.argv[1]);arm=sys.argv[2];out=ROOT/'results'/arm
REV='3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268'
class ResourceConflict(RuntimeError):pass
def read(p):return json.loads(Path(p).read_text())
def load_student(pipe,arm,manifest):
 ck=Path(manifest['models'][arm]['checkpoint'])
 shards=[torch.load(ck/f'student_rank{i}.pt',map_location='cpu',mmap=True,weights_only=True) for i in range(4)]
 targets=pipe.unet.state_dict()
 assert all(s.keys()==targets.keys() for s in shards)
 for k,target in targets.items():
  pieces=[s[k] for s in shards]
  if pieces[0].shape==target.shape and all(t.shape==target.shape for t in pieces):
   assert all(torch.equal(t,pieces[0]) for t in pieces[1:]),k
   value=pieces[0]
  else:
   value=torch.cat(pieces,0)
  assert value.shape==target.shape and torch.isfinite(value).all(),k
  target.copy_(value)
 print('STRICT_STUDENT_LOADED',arm,len(targets),flush=True)
core.load_student=load_student
def unit(t):return torch.nn.functional.normalize(t.float(),dim=-1).cpu().numpy()
def main():
 torch.set_num_threads(2);torch.set_num_interop_threads(1)
 manifest=read(out/'manifest.json');selection=read(ROOT/'selection.json');cases=selection['cases'];ids=[r['index'] for r in cases];chosen=set(ids)
 assert len(chosen)==1024 and selection['revision']==REV
 assert [dict(index=i,seed=manifest['samples'][i]['seed'],prompt=manifest['samples'][i]['prompt']) for i in ids]==cases
 ck=Path(manifest['models'][arm]['checkpoint']);record=manifest['models'][arm]
 assert (ck/'complete.json').is_file() and record['world_size']==4
 core.ROOT=ROOT;core.BASE=ROOT/'models/sdxl-base-1.0';core.OUT=out;core.RUNS=manifest['models'];core.METHODS=[arm]
 md=ROOT/'models/clip';assert (md/'revision.txt').read_text().strip()==REV
 model=CLIPModel(CLIPConfig.from_pretrained(md,local_files_only=True)).eval()
 state=torch.load(md/'pytorch_model.bin',weights_only=True,map_location='cpu')
 for k in ('vision_model.embeddings.position_ids','text_model.embeddings.position_ids'):
  if k not in model.state_dict():state.pop(k,None)
 model.load_state_dict(state,strict=True);del state
 tok=CLIPTokenizer.from_pretrained(md,local_files_only=True);proc=CLIPImageProcessor.from_pretrained(md,local_files_only=True)
 clipdir=out/'clip_features';clipdir.mkdir(exist_ok=True)
 original_generate=core.generate
 @torch.inference_mode()
 def generate(pipe,method,prompts,seeds):
  images,info=original_generate(pipe,method,prompts,seeds)
  pick=[(j,s-core.SEED) for j,s in enumerate(seeds) if s-core.SEED in chosen and not (clipdir/f'{s-core.SEED:05d}.npy').exists()]
  if pick:
   x=proc(images=[images[j] for j,i in pick],return_tensors='pt')['pixel_values']
   z=unit(model.visual_projection(model.vision_model(pixel_values=x).pooler_output))
   assert np.isfinite(z).all()
   for (j,i),v in zip(pick,z):
    p=clipdir/f'{i:05d}.npy'
    with p.with_suffix('.partial').open('wb') as f:np.save(f,v)
    p.with_suffix('.partial').replace(p)
  return images,info
 core.generate=generate
 args=SimpleNamespace(rank=0,world=1,limit=0,probe=False,methods=arm,batch=2,teacher_batch=2)
 core.worker(args)
 iv=np.stack([np.load(clipdir/f'{i:05d}.npy') for i in ids]);assert iv.shape==(1024,512) and np.isfinite(iv).all()
 tv=[]
 with torch.inference_mode():
  for j in range(0,1024,16):
   x=tok([r['prompt'] for r in cases[j:j+16]],padding=True,truncation=True,max_length=77,return_tensors='pt')
   tv.append(unit(model.text_projection(model.text_model(**x).pooler_output)))
 sim=iv@np.concatenate(tv).T;pos=sim.diagonal()*100;prompts=[r['prompt'] for r in cases]
 ranks=np.array([1+np.count_nonzero(sim[i]>max(sim[i,j] for j,t in enumerate(prompts) if t==p)) for i,p in enumerate(prompts)])
 core.write_json(out/'clip_results.json',dict(n=1024,revision=REV,cases=cases,metrics={arm:dict(cosine_x100=float(pos.mean()),top1_correct=int((ranks==1).sum()),top5_correct=int((ranks<=5).sum()))},per_case={arm:dict(alignment=pos.tolist(),rank=ranks.tolist())}))
 del model;gc.collect();torch.cuda.empty_cache()
 core.metrics()
 scores=read(out/'scores.json')[arm];assert scores['n']==10000 and all(np.isfinite(scores[k]) for k in ('FID','Patch_FID'))
 core.write_json(out/'all_complete.json',dict(arm=arm,FID=scores['FID'],Patch_FID=scores['Patch_FID'],CLIP=float(pos.mean()),fid_n=10000,clip_n=1024,finished_unix=time.time()))
 print('ALL_METRICS_COMPLETE',arm,flush=True)
if __name__=='__main__':
 try:main()
 except ResourceConflict as e:print('RESOURCE_CONFLICT',str(e),flush=True);sys.exit(75)
