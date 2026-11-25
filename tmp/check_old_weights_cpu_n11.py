from pathlib import Path
import json,torch,gc
from diffusers import UNet2DConditionModel
torch.set_num_threads(2)
r=Path('/data1/shenglin.zeng/coco_missing_20261111');specs=json.loads((r/'checkpoint_specs.json').read_text())
results={}
for name,relative in specs.items():
 if name=='A_pcm10_mix1':continue
 p=r/relative;meta=json.loads((p/'complete.json').read_text());shapes=json.loads((p/'shapes_rank0.json').read_text())
 shards=[torch.load(p/f'student_rank{i}.pt',map_location='cpu',weights_only=True,mmap=True) for i in range(meta['world_size'])]
 state={}
 for k,v in shapes.items():
  if v['placements'] and v['placements'][0].startswith('S('):value=torch.cat([s[k] for s in shards],dim=int(v['placements'][0][2:-1]))
  else:
   value=shards[0][k]
   assert all(torch.equal(value,s[k]) for s in shards[1:]),k
  assert list(value.shape)==v['shape'] and torch.isfinite(value).all(),k
  state[k]=value
 with torch.device('meta'):net=UNet2DConditionModel.from_config(json.loads((r/'models/sdxl-base-1.0/unet/config.json').read_text()))
 net.load_state_dict(state,strict=True,assign=True)
 results[name]={'strict_load':True,'finite':True,'world_size':meta['world_size'],'tensor_count':len(state)}
 print('CPU_LOAD_OK',name,results[name],flush=True)
 del net,state,shards;gc.collect()
(r/'old_checkpoints_cpu_validation.json').write_text(json.dumps(results,indent=2))
