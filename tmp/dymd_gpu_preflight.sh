set -eu
B=/data1/shenglin.zeng/dymd_prior_8_20261101
test -f "$B/ENV_READY"
"$B/env/bin/python" - <<'PY'
import subprocess
r=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
for row in r.splitlines():
 i,m,u=map(int,row.split(','))
 if i in [1,4,5,7]:assert m<500 and u==0,(i,m,u)
PY
export CUDA_VISIBLE_DEVICES=1,4,5,7
export NCCL_SOCKET_IFNAME=enp129s0f0 GLOO_SOCKET_IFNAME=enp129s0f0 NCCL_IB_DISABLE=1 OMP_NUM_THREADS=2
"$B/env/bin/torchrun" --nproc-per-node=4 --standalone "$B/gpu_preflight.py" > "$B/gpu_preflight.log" 2>&1
cat "$B/gpu_preflight.log"
