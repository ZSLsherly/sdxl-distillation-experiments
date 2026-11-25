set -euo pipefail
cd /data1/shenglin.zeng/sdxl_eval_uniform_20261025
export LD_LIBRARY_PATH="$PWD/runtime/lib:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES=''
env/bin/python - <<'PY'
import torch,diffusers,transformers,json
from pathlib import Path
r=Path.cwd()
print('versions',torch.__version__,diffusers.__version__,transformers.__version__)
for x in ['models/sdxl-base-1.0/unet/config.json','datasets/parallel_10k_seed20261018/prepared.pt']:
 p=r/x; assert p.is_file(),p; print(x,p.stat().st_size)
p=r/'outputs/turning_boundary_no_offload_2gpu_20261027'
p.mkdir(exist_ok=False)
(p/'code').mkdir()
PY
tar -xzf /data1/shenglin.zeng/turning_no_offload_20261027.tar.gz -C outputs/turning_boundary_no_offload_2gpu_20261027/code
cp /data1/shenglin.zeng/run_turning_no_offload_4090_20261027.py outputs/turning_boundary_no_offload_2gpu_20261027/workflow.py
unset CUDA_VISIBLE_DEVICES
nohup env/bin/python -u outputs/turning_boundary_no_offload_2gpu_20261027/workflow.py > outputs/turning_boundary_no_offload_2gpu_20261027/workflow.log 2>&1 < /dev/null &
echo WORKFLOW_PID=$!
