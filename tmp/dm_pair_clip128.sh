set -eu
. /root/sdxl_research/env.sh
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 HF_HUB_OFFLINE=1
python - <<'PY'
from pathlib import Path
import os,json,torch
import numpy as np
from PIL import Image
from transformers import CLIPConfig,CLIPModel,CLIPTokenizer,CLIPImageProcessor
work=Path(os.environ['SDXL_WORK']); root=Path(os.environ['SDXL_ROOT'])/'outputs'
d=work/'semantic_abc_20261103/model'; old=json.loads((work/'semantic_four_1024_20261103/results.json').read_text())
cases=old['cases'][:128];ids=[r['index'] for r in cases]; prompts=[r['prompt'] for r in cases]
torch.set_num_threads(2)
with torch.inference_mode():
 model=CLIPModel(CLIPConfig.from_pretrained(d)).eval()
 sd=torch.load(d/'pytorch_model.bin',map_location='cpu',weights_only=True)
 for k in ('vision_model.embeddings.position_ids','text_model.embeddings.position_ids'):
  if k not in model.state_dict():sd.pop(k,None)
 model.load_state_dict(sd,strict=True); del sd
 proc=CLIPImageProcessor.from_pretrained(d); tok=CLIPTokenizer.from_pretrained(d)
 unit=lambda x:torch.nn.functional.normalize(x.float(),dim=-1)
 text=torch.cat([unit(model.text_projection(model.text_model(**tok(prompts[j:j+16],padding=True,truncation=True,max_length=77,return_tensors='pt')).pooler_output)) for j in range(0,128,16)])
 vals={k:v['alignment'][:128] for k,v in old['per_case'].items()}
 for arm in ('DM234_fake3','DM1_mid4_fake3'):
  folder=root/'c_mix1_dm_pair_coco10k'/arm
  m=json.loads((folder/'manifest.json').read_text())
  assert m['visual_indices']==ids and all((m['samples'][r['index']]['prompt'],m['samples'][r['index']]['seed'])==(r['prompt'],r['seed']) for r in cases)
  feats=[]
  for j in range(0,128,8):
   ims=[Image.open(folder/'images'/arm/f'{i:05d}.png').convert('RGB') for i in ids[j:j+8]]
   x=proc(images=ims,return_tensors='pt').pixel_values
   feats.append(unit(model.visual_projection(model.vision_model(pixel_values=x).pooler_output)))
  vals[arm]=((torch.cat(feats)@text.T).diag()*100).tolist()
 result={k:float(np.mean(v)) for k,v in vals.items()}; diffs={}
 for arm in ('DM234_fake3','DM1_mid4_fake3'):
  v=np.array(vals[arm])-vals['C_mix1'];draw=np.random.default_rng(20261103).integers(128,size=(20000,128))
  diffs[arm]=[v.mean(),*np.quantile(v[draw].mean(1),[.025,.975])]
 out=dict(n=128,metrics=result,delta_vs_C_mix1=diffs,cases=cases,per_case=vals)
 (root/'c_mix1_dm_pair_coco10k/clip128.json').write_text(json.dumps(out))
 print(result,diffs)
PY
