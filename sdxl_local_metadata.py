import sys
sys.path.insert(0,'.image_runtime_pkgs')
import requests,json
from pathlib import Path
r=requests.get('https://huggingface.co/api/models/stabilityai/stable-diffusion-xl-base-1.0',timeout=45);r.raise_for_status();info=r.json()
Path('sdxl_model_info.json').write_text(json.dumps(info,indent=2))
print(info['sha']);print([x['rfilename'] for x in info['siblings']])
