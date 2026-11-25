set -eu
source /data5/shenglin.zeng/env.sh
cd /data5/shenglin.zeng/sdxl_crossdistill
nohup python -u - <<'PY' > outputs/coco10k_g160_20261020/inception_download.log 2>&1 &
import urllib.request,hashlib
from pathlib import Path
p=Path('models/metrics/clean_fid/inception-2015-12-05.pt')
p.parent.mkdir(parents=True,exist_ok=True)
if not p.exists():
    url='https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt'
    with urllib.request.urlopen(url,timeout=45) as response,p.with_suffix('.download').open('wb') as f:
        while b:=response.read(1<<20):f.write(b)
    p.with_suffix('.download').replace(p)
print('INCEPTION_DOWNLOADED',p.stat().st_size,hashlib.sha256(p.read_bytes()).hexdigest(),flush=True)
PY
echo "METRIC_DOWNLOAD_PID $!"
