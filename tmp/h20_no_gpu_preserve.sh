set -eu
cd /root/autodl-tmp/h20_direct_after_coco_20261103
/root/sdxl_research/env/bin/python - <<'PY'
from pathlib import Path
import hashlib,json,tarfile
p=Path('after_coco.train_only.upload')
s=p.read_text(encoding='utf-8-sig');compile(s,str(p),'exec')
assert 'DIRECT_G160_COMPLETE' in s and "original_script=" not in s
p.replace('after_coco.py')
root=Path('/root/autodl-tmp/sdxl_runs/outputs')
base=root/'h20_C_mix1_terminal_GAN_G160'
scores=root/(base.name+'_coco10k')
m=json.loads((scores/'manifest.json').read_text())['models']['C_mix1_terminal_GAN']
w=next((base/'weights').glob('*/student_rank0.pt'))
with w.open('rb') as f:actual=hashlib.file_digest(f,'sha256').hexdigest()
assert actual==m['shards'][0]['sha256']
d=root/'h20_C_mix1_direct_terminal_GAN_G160'
last=next((d/'eval').glob('*_g000032_periodic'))
with tarfile.open('/root/autodl-tmp/h20_direct_G33_evidence.tar.gz','w:gz',compresslevel=1) as tar:
 for n in ('config.json','data_manifest.json','rank0.jsonl','status.json'):
  tar.add(d/n,arcname='direct/'+n)
 tar.add(last,arcname='direct/G32_eval')
 tar.add(scores/'complete.json',arcname='refinement/coco10k_complete.json')
 tar.add(w.parent/'complete.json',arcname='refinement/weights_complete.json')
print('G160 SHA256',actual,'G32 png',len(list(last.glob('*.png'))),'queue train only')
PY
