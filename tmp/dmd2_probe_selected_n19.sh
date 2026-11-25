set -e
R=/data1/shenglin.zeng/dmd2_matched_10k_20261119
cd "$R/code"
"$R/env/bin/python" - <<'PY'
import json
from run_guard import check_idle
c=json.load(open('../config.json'));check_idle(c['gpu_uuids']);print('FOUR_SELECTED_GPUS_IDLE')
PY
export CUDA_VISIBLE_DEVICES=GPU-3a1e6a0e-9fab-18ef-c543-30066e01e383,GPU-dee80cca-d29a-9086-d006-b76ba6ee093e,GPU-4fba4d7f-66e5-aa77-7210-3803fe7190df,GPU-baca962f-0eff-b198-1677-8c28d525ecc1
export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 OMP_NUM_THREADS=2
"$R/env/bin/python" -m torch.distributed.run --standalone --nproc_per_node=4 "$R/code/gpu_probe.py"
