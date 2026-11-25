"""Fixed COCO1024 CLIP supplement. Images reused; generated pixels not stored."""
import os,sys,json,gc,time,fcntl,hashlib,subprocess
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from transformers import CLIPConfig,CLIPModel,CLIPTokenizer,CLIPImageProcessor
import evaluation_core as core
ROOT=Path('/data3/shenglin.zeng/clip_missing_20261113')
MD=ROOT/'models/clip'
REV='3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268'
class ResourceConflict(RuntimeError):pass
def read(p):return json.loads(Path(p).read_text())
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 q=p.with_suffix('.tmp');q.write_text(json.dumps(v,ensure_ascii=False,indent=2));q.replace(p)
def check_gpu():
 uuid=os.environ['CUDA_VISIBLE_DEVICES']
 assert uuid in ('GPU-5eceb5c0-44bb-9b56-d40e-2846ce7ff00f',)
 out=subprocess.check_output(['nvidia-smi','-i',uuid,'--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True,timeout=15)
 foreign={int(v.strip()) for v in out.splitlines() if v.strip().isdigit()}-{os.getpid()}
 if foreign:raise ResourceConflict('RESOURCE_CONFLICT: '+str(sorted(foreign)))
def load_clip():
 assert (MD/'revision.txt').read_text().strip()==REV
 model=CLIPModel(CLIPConfig.from_pretrained(MD,local_files_only=True)).eval()
 state=torch.load(MD/'pytorch_model.bin',weights_only=True,map_location='cpu')
 for k in ('vision_model.embeddings.position_ids','text_model.embeddings.position_ids'):
  if k not in model.state_dict():state.pop(k,None)
 model.load_state_dict(state,strict=True)
 return model,CLIPTokenizer.from_pretrained(MD,local_files_only=True),CLIPImageProcessor.from_pretrained(MD,local_files_only=True)
def unit(t):return torch.nn.functional.normalize(t.float(),dim=-1).cpu().numpy()
@torch.inference_mode()
def text_features(model,tok,prompts):
 chunks=[]
 for left in range(0,len(prompts),16):
  x=tok(prompts[left:left+16],padding=True,truncation=True,max_length=77,return_tensors='pt')
  chunks.append(unit(model.text_projection(model.text_model(**x).pooler_output)))
 return np.concatenate(chunks)
@torch.inference_mode()
def image_features(model,proc,images):
 x=proc(images=images,return_tensors='pt')['pixel_values']
 return unit(model.visual_projection(model.vision_model(pixel_values=x).pooler_output))
def load_student(pipe,arm,manifest):
 rec=manifest['models'][arm];p=Path(rec['checkpoint'])
 if (p/'shapes_rank0.json').exists():return original_load_student(pipe,arm,manifest)
 # Two archived single-rank checkpoints lack the separate shape sidecar.
 # Strict model loading checks every actual parameter name and tensor shape.
 assert rec['world_size']==1
 state=torch.load(p/'student_rank0.pt',map_location='cpu',weights_only=True,mmap=True)
 pipe.unet.load_state_dict(state,strict=True)
original_load_student=core.load_student
core.load_student=load_student
@torch.inference_mode()
def validate():
 known=read(ROOT/'existing/old_uniform.json');cases=known['cases'][:128]
 model,tok,proc=load_clip();tv=text_features(model,tok,[x['prompt'] for x in cases]);iv=[]
 for j in range(0,128,8):
  ims=[Image.open(ROOT/f"validation_images/{r['index']:05d}.png").convert('RGB') for r in cases[j:j+8]]
  iv.append(image_features(model,proc,ims))
 values=(np.concatenate(iv)*tv).sum(1)*100
 old=np.asarray(known['per_case']['joint4_uniform']['alignment'][:128])
 error=float(np.max(np.abs(values-old)));assert error<.015,error
 write(ROOT/'validation.json',dict(n=128,clip_revision=REV,max_abs_error=error,passed=True))
 print('VALIDATED',error,flush=True)
@torch.inference_mode()
def run(arm):
 out=ROOT/'arms'/arm;start=time.monotonic()
 manifest=read(out/'manifest.json');reference=read(ROOT/'selection.json')
 ids=[c['index'] for c in reference['cases']];prompts=[c['prompt'] for c in reference['cases']]
 assert len(set(ids))==1024 and reference['revision']==REV
 assert [dict(index=i,seed=manifest['samples'][i]['seed'],prompt=manifest['samples'][i]['prompt']) for i in ids]==reference['cases']
 ready=read(out/'ready.json')
 for s in ready['shards']:
  with (out/'weights'/s['name']).open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==s['sha256']
 model,tok,proc=load_clip();tv=text_features(model,tok,prompts)
 core.BASE=ROOT/'models/sdxl-base-1.0';core.RUNS=manifest['models']
 pipe=None;parts=[];generated=0;reused=0
 (out/'features').mkdir(exist_ok=True)
 for j in range(0,1024,64):
  p=out/f'features/{j:04d}.npz';chunk=ids[j:j+64]
  if not p.exists():
   feats=[]
   for k in range(0,64,2):
    check_gpu();batch=chunk[k:k+2];ims=[None]*len(batch);missing=[]
    for b,idx in enumerate(batch):
     cache=out/f'images/{idx:05d}.png'
     if cache.exists():
      with Image.open(cache) as im:ims[b]=im.convert('RGB')
      reused+=1
     else:missing.append(b)
    if missing:
     if pipe is None:check_gpu();pipe=core.load_pipeline(arm,manifest)
     rows=[manifest['samples'][batch[b]] for b in missing]
     new,_=core.generate(pipe,arm,[r['prompt'] for r in rows],[r['seed'] for r in rows])
     for b,im in zip(missing,new):ims[b]=im
     generated+=len(missing)
    feats.append(image_features(model,proc,ims))
   with p.with_suffix('.tmp').open('wb') as f:np.savez(f,indices=chunk,features=np.concatenate(feats))
   p.with_suffix('.tmp').replace(p)
  with np.load(p) as z:
   assert z['indices'].tolist()==chunk and z['features'].shape==(64,512) and np.isfinite(z['features']).all()
   parts.append(z['features'].copy())
  write(out/'status.json',dict(arm=arm,done=j+64,n=1024,seconds=time.monotonic()-start,generated_this_run=generated,reused_this_run=reused))
  print('BLOCK',arm,j+64,flush=True)
 im=np.concatenate(parts);sim=im@tv.T;pos=sim.diagonal()*100
 rank=np.array([1+np.count_nonzero(sim[i]>max(sim[i,j] for j,t in enumerate(prompts) if t==p)) for i,p in enumerate(prompts)])
 result=dict(n=1024,revision=REV,model='openai/clip-vit-base-patch32',cases=reference['cases'],
  metrics={arm:dict(cosine_x100=float(pos.mean()),top1_correct=int((rank==1).sum()),top5_correct=int((rank<=5).sum()))},
  per_case={arm:dict(alignment=pos.tolist(),rank=rank.tolist())},weights=ready['shards'],seconds=time.monotonic()-start)
 write(out/'results.json',result);print('COMPLETE',arm,result['metrics'],flush=True)
if __name__=='__main__':
 torch.set_num_threads(2);torch.set_num_interop_threads(1)
 if sys.argv[1]=='validate':validate()
 else:
  try:run(sys.argv[1])
  except ResourceConflict as error:
   print(str(error),flush=True);sys.exit(75)
