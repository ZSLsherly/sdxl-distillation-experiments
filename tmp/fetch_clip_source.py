from pathlib import Path
import sys
sys.path.insert(0,'.image_runtime_pkgs')
import requests
r=requests.get('https://raw.githubusercontent.com/openai/CLIP/main/clip/model.py',timeout=45);r.raise_for_status()
Path('tmp/sdxl_reference/openai_clip_model.py').write_text(r.text,encoding='utf-8')
print('CLIP reference source saved',len(r.content))
