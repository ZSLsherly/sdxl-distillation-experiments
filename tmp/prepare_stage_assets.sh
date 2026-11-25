set -euo pipefail
source /data5/shenglin.zeng/env.sh
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
root=/data5/shenglin.zeng/sdxl_crossdistill
mkdir -p "$root/exports/stage_probe_20261026/packages"
python -m pip install --no-deps --target "$root/exports/stage_probe_20261026/packages" lpips==0.1.4
export PYTHONPATH="$root/exports/stage_probe_20261026/packages"
python -u - <<'PY'
import lpips,torch,json
from pathlib import Path
m=lpips.LPIPS(net='squeeze').eval()
root=Path('/data5/shenglin.zeng/sdxl_crossdistill');out=root/'exports/stage_probe_20261026'
torch.save(m.state_dict(),out/'lpips_squeeze.pt')
ck=root/'outputs/joint4_uniform_10k_2gpu_mb16_20261023/train/weights/20261024T063356+0800_g000160_periodic'
assert json.loads((ck/'complete.json').read_text())['student_updates']==160
print('READY',ck,(out/'lpips_squeeze.pt').stat().st_size,flush=True)
PY
