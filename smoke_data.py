import os,io,tarfile,json,time,gc
from pathlib import Path
import torch
from imaginaire.utils import distributed
ROOT=Path('/data5/shenglin.zeng'); MODEL=ROOT/'models/Wan2.1-T2V-1.3B'
prompts=['A red panda walking slowly through a green bamboo forest.','Ocean waves rolling onto a sandy beach at sunset.','A small yellow bird sitting on a flowering tree branch.','A steam train traveling through a mountain valley.']
distributed.init()
rank=distributed.get_rank()
if os.environ.get('PREP_STAGE')=='embeddings':
 from sparkdiffusion.utils.umt5 import get_umt5_embedding
 from sparkdiffusion.datasets.build_synthetic_dataset import _DEFAULT_NEGATIVE_PROMPT
 embeddings=get_umt5_embedding(str(MODEL/'models_t5_umt5-xxl-enc-bf16.pth'),prompts+[_DEFAULT_NEGATIVE_PROMPT],tokenizer_path=str(MODEL/'google/umt5-xxl'))
 torch.save(embeddings[:4].cpu(),ROOT/'datasets/smoke_embeddings.pt')
 torch.save(embeddings[4].cpu(),ROOT/'models/negative_embed.pt')
 print('EMBEDDINGS_COMPLETE',flush=True)
else:
 from types import SimpleNamespace
 from sparkdiffusion.datasets.build_synthetic_dataset import _load_dit,sample_latents_from_prompts,write_to_tar
 net=_load_dit('1.3B',str(MODEL))
 emb=torch.load(ROOT/'datasets/smoke_embeddings.pt',weights_only=True)[rank:rank+1].cuda()
 neg=torch.load(ROOT/'models/negative_embed.pt',weights_only=True).cuda()
 if neg.ndim==2:neg=neg.unsqueeze(0)
 tokenizer=SimpleNamespace(latent_ch=16,spatial_compression_factor=8,get_latent_num_frames=lambda n:(n-1)//4+1)
 args=SimpleNamespace(resolution='480p',aspect_ratio='16:9',num_frames=81,seed=42,sigma_max=5000,timestep_shift=5.,sampler='Euler',num_steps=8,model_size='1.3B',boundary_ratio=.875,guidance_scale_high=5.,guidance_scale=5.)
 start=time.perf_counter()
 latent=sample_latents_from_prompts([rank],emb,neg,{'high':net,'low':net},tokenizer,args).cpu()[0]
 assert torch.isfinite(latent).all()
 folder=ROOT/'datasets/smoke_1.3B_480p_81f_8step';folder.mkdir(exist_ok=True)
 with tarfile.open(folder/f'shard_{rank:06d}.tar','w') as tar:
  for rep in range(16):
   prefix=f'{rank*16+rep:09d}'
   for name,tensor in [('latent',latent),('embed',emb.cpu()[0])]:
    b=io.BytesIO();torch.save(tensor,b);write_to_tar(tar,f'{prefix}.{name}.pt',b.getvalue())
   write_to_tar(tar,f'{prefix}.prompt.txt',prompts[rank].encode())
 print(json.dumps({'rank':rank,'latent_shape':list(latent.shape),'seconds':time.perf_counter()-start,'peak_GiB':torch.cuda.max_memory_allocated()/1024**3}),flush=True)
 print('SHARD_COMPLETE',rank,flush=True)
torch.distributed.destroy_process_group()
