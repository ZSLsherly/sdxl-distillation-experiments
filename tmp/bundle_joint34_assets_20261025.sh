set -euo pipefail
source /data5/shenglin.zeng/env.sh
export CUDA_VISIBLE_DEVICES=''
cd /data5/shenglin.zeng/sdxl_crossdistill
nice -n 19 ionice -c 2 -n 7 python - <<'PY'
from pathlib import Path
import json,hashlib,tarfile,shutil
root=Path.cwd();out=root/'exports/joint34_uniform_20261025';out.mkdir(parents=True,exist_ok=False)
ref=root/'outputs/joint4_uniform_10k_2gpu_mb16_20261023/train'
source=ref.parent/'code/sdxl_research';hashes=json.loads((ref/'source_sha256.json').read_text())
assert all(hashlib.sha256((source/n).read_bytes()).hexdigest()==v for n,v in hashes.items())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
data=root/'datasets/parallel_10k_seed20261018/prepared.pt'
provenance={'prepared_pt_sha256':sha(data),'source_sha256':hashes,'reference_config':json.loads((ref/'config.json').read_text()),'initial_config':json.loads((root/'outputs/joint4_uniform_10k_2gpu_20261023/train/config.json').read_text())}
(out/'provenance.json').write_text(json.dumps(provenance,indent=2))
with tarfile.open(out/'training_assets.tar','w') as t:
 for rel in ['datasets/parallel_10k_seed20261018','models/dinov2-small']:t.add(root/rel,arcname=rel)
 for p in source.glob('*.py'):t.add(p,arcname='training_code/'+p.name)
 t.add(out/'provenance.json',arcname='joint34_reference_provenance.json')
meta={'training_assets.tar':{'bytes':(out/'training_assets.tar').stat().st_size,'sha256':sha(out/'training_assets.tar')}}
(out/'transfer_manifest.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta),flush=True)
PY
