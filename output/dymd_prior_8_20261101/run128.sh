#!/bin/bash
set -euo pipefail
B=/data1/shenglin.zeng/dymd_prior_8_20261101
cd "$B/official/pf_wan"
export PATH="$B/env/bin:$PATH"
export PYTHONPATH="$PWD"
export CUDA_VISIBLE_DEVICES=${DYMD_GPU_IDS:-1,2,3,4}
export DYMD_PROFILE_DIR="$B/profiles"
export DYMD_SAMPLE_PLAN="$B/sample_plan.json"
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export NCCL_SOCKET_IFNAME=enp129s0f0 GLOO_SOCKET_IFNAME=enp129s0f0 NCCL_IB_DISABLE=1 NCCL_DEBUG=WARN
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python "$B/prepare_run.py"
python - <<'PY'
import subprocess,csv,io,os
selected=[int(x) for x in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]
rows=list(csv.reader(io.StringIO(subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True))))
for i,mem,util in rows:
 if int(i) in selected: assert int(mem)<500 and int(util)==0,('GPU no longer idle',i,mem,util)
PY
if [ -s assets/examples/prior128.txt ]; then
 torchrun --nproc-per-node=4 --nnodes=1 --rdzv-endpoint=127.0.0.1:29631 "$B/profile_official.py" --config-file configs/generate/prior128.jsonc
fi
touch "$B/RUN_COMPLETE"
