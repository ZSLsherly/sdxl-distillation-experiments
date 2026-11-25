"""Save official pinned source metadata, without downloading large data locally."""
import json, urllib.request
from pathlib import Path

REV = 'be22767697a1f3ca656b73c776e15fa335c86c6c'
FOLDER = 'data/laion_vae_latents/sdxl_vae_latents_laion_500k'
url = f'https://huggingface.co/api/models/tianweiy/DMD2/tree/{REV}/{FOLDER}?recursive=false&expand=false'
with urllib.request.urlopen(url, timeout=30) as response:
    rows = json.load(response)
selected = []
for i in range(2, 9):
    name = f'vae_latents_{i:03d}.pt'
    row = next(r for r in rows if r['path'].endswith('/'+name))
    selected.append(dict(name=name, size=row['size'], sha256=row['lfs']['oid'],
                         url=f'https://huggingface.co/tianweiy/DMD2/resolve/{REV}/{FOLDER}/{name}?download=true',
                         shard=i, fallback_only=(i == 8)))
out = Path('output/sdxl-crossdistill/a50k_dataset_20261123')
out.mkdir(parents=True, exist_ok=True)
(out/'sources.json').write_text(json.dumps(dict(repo='tianweiy/DMD2',revision=REV,files=selected), indent=2), encoding='utf-8')
print('PINNED',len(selected),'shards; six required, one fallback; no large local download')
