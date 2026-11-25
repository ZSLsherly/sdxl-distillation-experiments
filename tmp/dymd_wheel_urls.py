import requests,re,json
from urllib.parse import unquote,urljoin
items=[]
for pkg in ['torch','torchvision','torchaudio']:
 base=f'https://download.pytorch.org/whl/cu128/{pkg}/'
 page=requests.get(base,timeout=30);page.raise_for_status()
 ver={'torch':'2.7.1','torchvision':'0.22.1','torchaudio':'2.7.1'}[pkg]
 urls=re.findall(r'href="([^"]+)"',page.text)
 matches=[urljoin(base,x) for x in urls if f'{pkg}-{ver}+cu128-cp311-cp311-manylinux_2_28_x86_64.whl' in unquote(x)]
 assert len(matches)==1,(pkg,matches)
 items.append(matches[0])
print(json.dumps(items))
from pathlib import Path
Path('output/dymd_prior_8_20261101/wheel_urls.json').write_text(json.dumps(items))
