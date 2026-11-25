"""Fixed COCO10K FID/PFID plus historical COCO1024 CLIP, official DMD2 sampler."""
import os,sys,json,time,gc,hashlib,fcntl,ctypes,argparse
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel,CLIPConfig,CLIPTokenizer,CLIPImageProcessor
import evaluation_core as core
from native_sample_reference import sample as official_sample

p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','probe','worker','metrics']);p.add_argument('--root',type=Path,required=True);p.add_argument('--rank',type=int,default=0);p.add_argument('--world',type=int,default=4);args=p.parse_args()
R=args.root;J=json.loads((R/'job.json').read_text());BASELINE=json.loads((R/'inputs/A_baseline.json').read_text());METHOD=J['method']
core.ROOT=Path(J['asset_root']);core.BASE=core.ROOT/'models/sdxl-base-1.0';core.OUT=R;core.RUNS={};core.METHODS=[METHOD]
torch.set_num_threads(2);torch.set_num_interop_threads(1)
torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
write=core.write_json

def idle():
 import pynvml as n
 n.nvmlInit()
 try:
  h=n.nvmlDeviceGetHandleByUUID(os.environ['CUDA_VISIBLE_DEVICES'])
  assert not [p for p in n.nvmlDeviceGetComputeRunningProcesses(h) if p.pid!=os.getpid()],'Selected GPU already occupied'
  assert n.nvmlDeviceGetMemoryInfo(h).free>30*1024**3 and n.nvmlDeviceGetUtilizationRates(h).gpu<=5
 finally:n.nvmlShutdown()

def clip_load():
 d=Path(J['clip_model']);assert (d/'revision.txt').read_text().strip()==J['clip_revision']
 m=CLIPModel(CLIPConfig.from_pretrained(d,local_files_only=True)).eval().requires_grad_(False)
 sd=torch.load(d/'pytorch_model.bin',map_location='cpu',weights_only=True)
 for k in ['vision_model.embeddings.position_ids','text_model.embeddings.position_ids']:
  if k not in m.state_dict():sd.pop(k,None)
 m.load_state_dict(sd,strict=True);del sd
 return m,CLIPTokenizer.from_pretrained(d,local_files_only=True),CLIPImageProcessor.from_pretrained(d,local_files_only=True)

def unit(v):return torch.nn.functional.normalize(v.float(),dim=-1).cpu().numpy()
@torch.inference_mode()
def clip_images(m,proc,images):
 x=proc(images=images,return_tensors='pt')['pixel_values']
 return unit(m.visual_projection(m.vision_model(pixel_values=x).pooler_output))
@torch.inference_mode()
def clip_texts(m,tok,texts):
 out=[]
 for j in range(0,len(texts),16):
  x=tok(texts[j:j+16],padding=True,truncation=True,max_length=77,return_tensors='pt')
  out.append(unit(m.text_projection(m.text_model(**x).pooler_output)))
 return np.concatenate(out)

def load_pipe():
 pipe=core.load_pipeline('base_only',{})
 state=torch.load(J['checkpoint'],map_location='cpu',weights_only=True,mmap=True)
 pipe.unet.load_state_dict(state,strict=True);del state;gc.collect();ctypes.CDLL('libc.so.6').malloc_trim(0)
 return pipe

def prepare():
 ref=json.loads((R/'inputs/reference_manifest.json').read_text());existing=json.loads((core.ROOT/'outputs/coco10k/manifest.json').read_text())
 assert ref['n']==10000 and ref['csv_sha256']=='fc1efb7dd0f7d561a999125340fc4b10632d28f2547c17b174170514889c8599'
 assert ref['samples']==existing['samples'] and ref['protocol']==existing['protocol'] and ref['visual_indices']==existing['visual_indices']
 cases=BASELINE['CLIP_cases'];assert len(cases)==1024 and len({x['index'] for x in cases})==1024
 assert cases==[dict(index=x['index'],seed=ref['samples'][x['index']]['seed'],prompt=ref['samples'][x['index']]['prompt']) for x in cases]
 assert BASELINE['CLIP_revision']==J['clip_revision']
 featurefiles=[];ids=[]
 for f in sorted(Path(J['reference_features']).glob('*.npz')):
  with np.load(f) as z:
   ii=z['indices'].tolist();ids.extend(ii)
   for key in ['full','patch']:assert z[key].shape==(len(ii),2048) and np.isfinite(z[key]).all()
  featurefiles.append(dict(path=str(f),sha256=core.sha(f)))
 assert ids==list(range(10000))
 inc=core.ROOT/'models/metrics/clean_fid/inception-2015-12-05.pt';assert core.sha(inc)=='f58cb9b6ec323ed63459aa4fb441fe750cfe39fafad6da5cb504a16f19e958f4'
 ck=Path(J['checkpoint']);completed=json.loads((ck.parent/'complete.json').read_text());assert completed['G']==160 and completed['F']==800 and completed['strict_load'] and completed['finite']
 manifest=dict(ref,models={METHOD:{'checkpoint':str(ck),'sha256':core.sha(ck),'G':160,'F':800}},CLIP_n=1024,CLIP_revision=J['clip_revision'],sampler='Official DMD2 x0 -> independent re-noise, 999/749/499/249, CFG1; independent fixed per-case stage seeds seed+100000*(stage+1), same as historical official DMD2 benchmark',reference_features=featurefiles)
 if (R/'manifest.json').exists():assert json.loads((R/'manifest.json').read_text())==manifest
 else:write(R/'manifest.json',manifest)
 dest=R/'features/real';dest.parent.mkdir(exist_ok=True)
 if not dest.exists():dest.symlink_to(J['reference_features'],target_is_directory=True)
 assert dest.resolve()==Path(J['reference_features']).resolve()
 write(R/'prepared.json',{'n':10000,'CLIP_n':1024,'reference_blocks':len(featurefiles),'checkpoint_sha256':manifest['models'][METHOD]['sha256']})
 print('PREPARED',flush=True)

@torch.inference_mode()
def probe():
 idle();m=json.loads((R/'manifest.json').read_text());clip,tok,proc=clip_load()
 known=json.loads(Path(J['validation_clip']).read_text());cases=known['cases'][:8]
 ims=[Image.open(Path(J['validation_images'])/f'{x["index"]:05d}.png').convert('RGB') for x in cases]
 iv=clip_images(clip,proc,ims);tv=clip_texts(clip,tok,[x['prompt'] for x in cases]);scores=(iv*tv).sum(1)*100
 old=np.array(known['per_case']['joint4_uniform']['alignment'][:8]);error=float(np.max(np.abs(scores-old)));assert error<.015,error
 del clip;gc.collect();pipe=load_pipe();sampler_errors=[]
 for idx in [0,124]:
  row=m['samples'][idx];ims,_=core.generate(pipe,'dmd2',[row['prompt']],[row['seed']])
  cond,_,pool,_=pipe.encode_prompt([row['prompt']],device='cuda',do_classifier_free_guidance=False)
  added={'text_embeds':pool,'time_ids':torch.tensor([[1024,1024,0,0,1024,1024]],device='cuda',dtype=torch.float32)}
  noise=torch.randn((1,4,128,128),device='cuda',generator=torch.Generator(device='cuda').manual_seed(row['seed']))
  noises=[torch.randn((1,4,128,128),device='cuda',generator=torch.Generator(device='cuda').manual_seed(row['seed']+100000*(j+1))) for j in range(4)]
  class AMP(torch.nn.Module):
   def forward(self,*a,**kw):
    with torch.autocast('cuda',dtype=torch.bfloat16):return pipe.unet(*a,**kw)
  scheduler=core.DDIMScheduler.from_pretrained(core.BASE,subfolder='scheduler',local_files_only=True)
  # Official reference inherits latent dtype from prompt_embed; FP32 matches its research script and our historical latent arithmetic.
  with patch('torch.randn_like',side_effect=noises):expected=official_sample(noise,added,AMP(),pipe.vae,scheduler,cond.float(),num_step=4)[0].cpu().numpy()
  diff=float(np.abs(np.asarray(ims[0],dtype=np.float32)-expected.astype(np.float32)).max());assert diff==0,diff;sampler_errors.append(diff)
  pd=R/'probe';pd.mkdir(exist_ok=True);ims[0].save(pd/f'{idx:05d}.png')
 inc=core.inception();full,patches=core.features(inc,ims);assert full.shape==patches.shape==(1,2048)
 write(R/'probe_passed.json',{'CLIP_reference_n':8,'CLIP_max_error':error,'official_sampler_pixel_max_errors':sampler_errors,'features_finite':True})
 print('PROBE_PASSED',flush=True)

@torch.inference_mode()
def worker():
 idle();assert (R/'probe_passed.json').exists();m=json.loads((R/'manifest.json').read_text());sel={x['index'] for x in BASELINE['CLIP_cases']};visual=set(m['visual_indices'])
 dest=R/'features'/METHOD;dest.mkdir(parents=True,exist_ok=True);imagesdir=R/'images';imagesdir.mkdir(exist_ok=True)
 # Avoid simultaneous full model construction exceeding shared host-memory limits.
 with (R/'load.lock').open('a') as gate:
  fcntl.flock(gate,fcntl.LOCK_EX);pipe=load_pipe();inc=core.inception();clip,_,proc=clip_load();gc.collect();ctypes.CDLL('libc.so.6').malloc_trim(0)
 batch=J['batch'];blocks=core.feature_blocks(10000,args.rank,args.world);target=sum(b-a for a,b in blocks);done=0;started=time.monotonic();torch.cuda.reset_peak_memory_stats()
 for left,right in blocks:
  file=dest/f'{left:05d}_{right:05d}.npz'
  if file.exists():
   with np.load(file) as z:
    assert z['indices'].tolist()==list(range(left,right))
    assert z['clip_indices'].tolist()==[i for i in range(left,right) if i in sel]
    for k in ['full','patch','clip']:assert np.isfinite(z[k]).all()
   done+=right-left;continue
  full=[];patches=[];cids=[];cv=[];at=left
  while at<right:
   ids=list(range(at,min(right,at+batch)));rows=[m['samples'][i] for i in ids]
   try:
    ims,_=core.generate(pipe,'dmd2',[x['prompt'] for x in rows],[x['seed'] for x in rows]);f,p=core.features(inc,ims)
   except torch.cuda.OutOfMemoryError:
    if batch==1:raise
    batch//=2;gc.collect();torch.cuda.empty_cache();print('REDUCE_BATCH',batch,flush=True);continue
   full.append(f);patches.append(p)
   chosen=[(i,im) for i,im in zip(ids,ims) if i in sel]
   if chosen:cids.extend(i for i,im in chosen);cv.append(clip_images(clip,proc,[im for i,im in chosen]))
   for i,im in zip(ids,ims):
    if i in visual:im.save(imagesdir/f'{i:05d}.png',compress_level=3)
   at+=len(ids)
  values=dict(indices=np.arange(left,right),full=np.concatenate(full),patch=np.concatenate(patches),clip_indices=np.array(cids,dtype=np.int64),clip=np.concatenate(cv) if cv else np.empty((0,512),dtype=np.float32))
  assert all(np.isfinite(v).all() for v in values.values())
  with file.with_suffix('.partial').open('wb') as f:np.savez(f,**values)
  file.with_suffix('.partial').replace(file);done+=right-left
  status=dict(rank=args.rank,done=done,target=target,batch=batch,seconds=time.monotonic()-started,peak_GiB=torch.cuda.max_memory_allocated()/2**30,updated=time.time())
  write(R/f'status_{args.rank}.json',status);print('BLOCK',json.dumps(status),flush=True)
 write(R/f'worker_{args.rank}_done.json',dict(done=done,time=time.time()));print('WORKER_DONE',args.rank,flush=True)

def metrics():
 core.metrics();(R/'complete.json').replace(R/'fid_complete.json')
 ids=[];feat=[]
 for p in sorted((R/'features'/METHOD).glob('*.npz')):
  with np.load(p) as z:ids.extend(z['clip_indices'].tolist());feat.extend(z['clip'])
 wanted=[x['index'] for x in BASELINE['CLIP_cases']];assert len(ids)==len(set(ids))==1024 and set(ids)==set(wanted)
 lookup=dict(zip(ids,feat));iv=np.stack([lookup[i] for i in wanted]);clip,tok,_=clip_load();texts=[x['prompt'] for x in BASELINE['CLIP_cases']];tv=clip_texts(clip,tok,texts);sim=iv@tv.T
 pos=sim.diagonal()*100;rank=np.array([1+np.count_nonzero(sim[i]>max(sim[i,j] for j,t in enumerate(texts) if t==p)) for i,p in enumerate(texts)])
 assert np.isfinite(pos).all();clip_result={'n':1024,'model':BASELINE['CLIP_model'],'revision':J['clip_revision'],'cosine_x100':float(pos.mean()),'top1':int((rank==1).sum()),'top5':int((rank<=5).sum()),'cases':BASELINE['CLIP_cases'],'alignment':pos.tolist()}
 write(R/'clip_scores.json',clip_result)
 scores=json.loads((R/'scores.json').read_text())[METHOD];assert scores['n']==10000
 values=dict(scores,CLIP=float(pos.mean()),CLIP_n=1024)
 delta={k:values[k]-BASELINE[k] for k in ['FID','Patch_FID','CLIP']}
 write(R/'comparison.json',{'A':{k:BASELINE[k] for k in ['FID','Patch_FID','CLIP','FID_n','CLIP_n']},METHOD:values,'DMD2_minus_A':delta})
 write(R/'complete.json',dict(completed=time.time(),scores=values,comparison=delta,protocol='FID/PFID10000; historical fixed CLIP1024; official stochastic DMD2'))
 print('ALL_COMPLETE',json.dumps(values),flush=True)

if __name__=='__main__':globals()[args.action]()
