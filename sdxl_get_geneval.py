import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,'.image_runtime_pkgs')
import requests
repo='djghosh13/geneval'
r=requests.get('https://api.github.com/repos/'+repo+'/commits/main',timeout=45);r.raise_for_status();commit=r.json()['sha']
url='https://raw.githubusercontent.com/'+repo+'/'+commit+'/prompts/evaluation_metadata.jsonl'
r=requests.get(url,timeout=45);r.raise_for_status()
rows=[json.loads(line) for line in r.text.splitlines() if line.strip()];assert all('prompt' in row for row in rows)
out=Path('output/sdxl-crossdistill');out.mkdir(parents=True,exist_ok=True)
(out/'geneval_evaluation_metadata.jsonl').write_bytes(r.content)
(out/'geneval_manifest.json').write_text(json.dumps({'repo':repo,'commit':commit,'url':url,'samples':len(rows),'sha256':hashlib.sha256(r.content).hexdigest()},indent=2))
print('OFFICIAL_GENEVAL_PROMPTS',len(rows),'revision',commit)
