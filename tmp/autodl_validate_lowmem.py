import hashlib, json, math, struct, sys, importlib.metadata as md
from pathlib import Path
root=Path('/root/autodl-tmp/sdxl_research')
import torch
from accelerate import init_empty_weights
from diffusers import UNet2DConditionModel,AutoencoderKL,DDIMScheduler
from torch.distributed._composable.fsdp import fully_shard,MixedPrecisionPolicy,CPUOffloadPolicy
print('IMPORTS_OK',flush=True)
manifest=json.loads((root/'transfers/manifest.json').read_text())
for name,path in [('prepared.pt',root/'datasets/parallel_10k_seed20261018/prepared.pt'),('dataset_manifest.json',root/'datasets/parallel_10k_seed20261018/manifest.json')]:
 h=hashlib.sha256()
 with path.open('rb') as f:
  for chunk in iter(lambda:f.read(4<<20),b''):h.update(chunk)
 assert path.stat().st_size==manifest[name]['bytes'] and h.hexdigest()==manifest[name]['sha256']
 print('DATA_HASH_OK',name,flush=True)
base=root/'models/sdxl-base-1.0'
verified={}
for folder,cls in [('unet',UNet2DConditionModel),('vae',AutoencoderKL)]:
 path=base/folder/'diffusion_pytorch_model.fp16.safetensors'
 with path.open('rb') as f:
  length=struct.unpack('<Q',f.read(8))[0];header=json.loads(f.read(length))
 tensors={k:v for k,v in header.items() if k!='__metadata__'}
 with init_empty_weights(include_buffers=True):
  model=cls.from_config(json.loads((base/folder/'config.json').read_text()))
 state=model.state_dict()
 assert set(state)==set(tensors),folder
 for key,value in state.items():assert list(value.shape)==tensors[key]['shape'],(folder,key)
 assert max(v['data_offsets'][1] for v in tensors.values())+8+length==path.stat().st_size
 verified[folder]={'tensors':len(tensors),'parameters':sum(p.numel() for p in model.parameters()),'bytes':path.stat().st_size,'shape_match':True}
 del model,state
 print('MODEL_HEADER_AND_CONFIG_OK',folder,verified[folder],flush=True)
scheduler=DDIMScheduler.from_pretrained(base,subfolder='scheduler',local_files_only=True)
d=torch.load(root/'datasets/parallel_10k_seed20261018/prepared.pt',weights_only=True,mmap=True,map_location='cpu')
assert d['train_count']==10000 and list(d['latents'].shape[1:])==[4,128,128]
shapes={k:list(v.shape) for k,v in d.items() if isinstance(v,torch.Tensor)}
for key,v in d.items():
 if isinstance(v,torch.Tensor) and v.is_floating_point() and v.ndim:
  for i in sorted(set([0,len(v)//2,len(v)-1])):
   assert torch.isfinite(v[i]).all().item(),(key,i)
print('DATA_METADATA_AND_SAMPLE_FINITE_OK',shapes,flush=True)
result={'status':'cpu_preparation_complete_gpu_validation_pending','train_count':10000,'tensor_shapes':shapes,'models':verified,'cuda_build':torch.version.cuda,'cuda_available':torch.cuda.is_available(),'packages':{p:md.version(p) for p in ['torch','torchvision','diffusers','transformers','accelerate','numpy','scipy','safetensors','Pillow']},'cpu_tests_passed':22,'full_model_load':'not_completed: earlier process killed; CPU container memory limit 2 GiB','model_validation':'meta-device architecture and all checkpoint tensor names/shapes; source archive SHA256 verified by setup','dataset_validation':'full SHA256 plus metadata and first/middle/last tensor sample finite checks','gpu_validation':'not_run','training_started':False}
(root/'preparation_complete.json').write_text(json.dumps(result,indent=2))
(root/'setup.status').write_text('cpu_ready_gpu_validation_pending\n')
print(json.dumps(result),flush=True)
