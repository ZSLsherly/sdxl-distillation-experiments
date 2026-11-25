set -eu
. /root/sdxl_research/env.sh
base=/root/autodl-tmp/h20_gan_mb32
c=/root/autodl-tmp/h20_gan_mb16/code
mkdir -p "$c"; cp "$base/code"/*.py "$base/code/config.json" "$c/"
cd "$c"
python - <<'PY'
import json
from pathlib import Path
p=Path('config.json');cfg=json.loads(p.read_text());cfg.update(batch=64,benchmark_batch=64,micro_batch=16);p.write_text(json.dumps(cfg,indent=2))
s=Path('run_h20.py').read_text().replace('h20_gan_mb32_a4','h20_gan_mb16_a4').replace("cfg['micro_batch']==32","cfg['micro_batch']==16").replace("cfg['benchmark_batch']==128","cfg['benchmark_batch']==64").replace('(3,6,3,32,4)','(3,6,3,16,4)')
Path('run_h20_16.py').write_text(s)
PY
L="$base/blas126/nvidia/cublas/lib";export LD_PRELOAD="$L/libcublasLt.so.12:$L/libcublas.so.12" PYTHONFAULTHANDLER=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
test ! -e "$SDXL_ROOT/outputs/h20_gan_mb16_a4"
nohup python -u run_h20_16.py </dev/null > h20.log 2>&1 & echo $!
