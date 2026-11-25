import sys,json
sys.path.insert(0,'.image_runtime_pkgs')
import requests
from pathlib import Path
for name,url in {
 'download_sdxl.sh':'https://raw.githubusercontent.com/tianweiy/DMD2/main/scripts/download_sdxl.sh',
 'dataset_api':'https://huggingface.co/api/datasets/tianweiy/DMD2',
 'model_api':'https://huggingface.co/api/models/tianweiy/DMD2',
}.items():
 r=requests.get(url,timeout=60)
 if r.status_code!=200:print(name,r.status_code);continue
 if name.endswith('api'):
  info=r.json();Path('sdxl_'+name+'.json').write_text(json.dumps(info,indent=2));print(name,info.get('sha'),[x['rfilename'] for x in info.get('siblings',[]) if any(p in x['rfilename'].lower() for p in ['laion','coco','prompt','dataset'])])
 else:print(name,r.text)
