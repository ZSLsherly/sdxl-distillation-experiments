set -eu
source /data5/shenglin.zeng/env.sh
python - <<'PY'
import sys,zipfile,io
from pathlib import Path
sys.path.insert(0,'/data5/shenglin.zeng/sdxl_crossdistill/src')
from sdxl_prepare import DataUnpickler
root=Path('/data5/shenglin.zeng/sdxl_crossdistill')
with zipfile.ZipFile(root/'datasets/dmd2/coco10k.zip') as archive:
 names=archive.namelist();print('COCO_PACKAGE',len(names),names[:6])
 member=next(n for n in names if n.endswith('all_prompts.pkl'))
 prompts=DataUnpickler(io.BytesIO(archive.read(member))).load()
 print('COCO_PROMPT_LIST',type(prompts),len(prompts),'EXAMPLE',str(prompts[:2])[:300])
assert isinstance(prompts,list) and len(prompts)==10000
PY
tail -n 6 /data5/shenglin.zeng/sdxl_crossdistill/logs/download.log
tail -n 6 /data5/shenglin.zeng/sdxl_crossdistill/logs/baselines_download.log
tail -n 12 /data5/shenglin.zeng/sdxl_crossdistill/logs/train.log
