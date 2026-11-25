set -eu
. /root/sdxl_research/env.sh
python - <<'PY'
from pathlib import Path
from PIL import Image
from io import BytesIO
import json, tarfile
r=Path('/root/sdxl_research/outputs/turning_BC_paired_quota_20261102')
arm='uniform_control'
with tarfile.open(r/'uniform_review.tar.gz','w:gz') as archive:
    for name in ('rank0.jsonl','config.json','status.json','source_sha256.json','data_manifest.json'):
        archive.add(r/arm/name,arcname=f'{arm}/{name}')
    archive.add(r/'checkpoint_validation_uniform_control.json',arcname='checkpoint_validation_uniform_control.json')
    for g in (0,40,80,118,120,160):
        d=next((r/arm/'eval').glob(f'*_g{g:06d}_*'))
        archive.add(d/'rank0.json',arcname=str((d/'rank0.json').relative_to(r)))
        if g!=160: continue
        for p in d.glob('*.png'):
            im=Image.open(p).convert('RGB')
            for tag,imv in [('full',im.resize((360,360),Image.Resampling.LANCZOS)),('crop',im.crop((362,362,661,661)))]:
                b=BytesIO(); imv.save(b,format='JPEG',quality=92); h=tarfile.TarInfo(f'preview/{p.stem}_{tag}.jpg'); h.size=b.tell(); b.seek(0); archive.addfile(h,b)
print((r/'uniform_review.tar.gz').stat().st_size)
PY
