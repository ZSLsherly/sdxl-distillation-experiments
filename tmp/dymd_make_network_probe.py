import requests,json
from urllib.parse import urlsplit
from pathlib import Path
url='https://huggingface.co/DAGroup-PKU/PF_Wan/resolve/8fa7e17d5f944d0275df7dc7929137b956367ff8/backbone.pth'
r=requests.head(url,allow_redirects=True,timeout=30);r.raise_for_status()
host=urlsplit(r.url).hostname
mapping={}
for h in [host,'huggingface.co','download.pytorch.org','download-r2.pytorch.org']:
 a=requests.get('https://dns.google/resolve',params={'name':h,'type':'A'},timeout=20).json()
 mapping[h]=[x['data'] for x in a.get('Answer',[]) if x['type']==1]
Path('tmp/dymd_network_probe.json').write_text(json.dumps({'url':r.url,'mapping':mapping}))
print('Resolved domains',list(mapping))
