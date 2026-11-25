set -eu
r=/data1/shenglin.zeng/sdxl_eval_uniform_20261025
cd "$r/outputs"
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 LD_LIBRARY_PATH="$r/runtime/lib:${LD_LIBRARY_PATH:-}"
"$r/env/bin/python" - <<'PY'
import json,torch,numpy as np
from pathlib import Path
from PIL import Image
from transformers import CLIPModel,CLIPConfig,CLIPTokenizer,CLIPImageProcessor
t=Path('semantic_joint4_uniform_1024');ref=Path('semantic_references_1024');m=t/'model'
torch.set_num_threads(2)
model=CLIPModel(CLIPConfig.from_pretrained(m)).eval()
sd=torch.load(m/'pytorch_model.bin',weights_only=True,map_location='cpu')
for k in ('vision_model.embeddings.position_ids','text_model.embeddings.position_ids'):
 if k not in model.state_dict():sd.pop(k,None)
model.load_state_dict(sd,strict=True);del sd
proc=CLIPImageProcessor.from_pretrained(m);tok=CLIPTokenizer.from_pretrained(m)
sel=np.sort(np.random.default_rng(42).choice(1024,32,replace=False))
for a in ['joint4_uniform','sdxl','dmd2']:
 old=json.loads((t/'results.json' if a=='joint4_uniform' else ref/(a+'_results.json')).read_text())
 iv=[];tv=[]
 with torch.inference_mode():
  for at in range(0,len(sel),8):
   ims=[];capt=[]
   for i in sel[at:at+8]:
    row=old['cases'][i];capt.append(row['prompt'])
    if a=='joint4_uniform':folder=(Path('coco10k/images')/a if i<128 else t/'images')
    else:folder=ref/('reference' if i<128 else 'images')/a
    with Image.open(folder/f"{row['index']:05d}.png") as im:ims.append(im.convert('RGB'))
   res=model(**proc(images=ims,return_tensors='pt'),**tok(capt,padding=True,truncation=True,max_length=77,return_tensors='pt'))
   iv.append(res.image_embeds);tv.append(res.text_embeds)
 s=(torch.cat(iv)@torch.cat(tv).T*100).numpy()
 print(a,'paired',s.diagonal().mean(),'mismatched',(s.sum()-s.trace())/(32*31),'max_delta',np.max(np.abs(s.diagonal()-np.array(old['per_case'][a]['alignment'])[sel])),flush=True)
PY
