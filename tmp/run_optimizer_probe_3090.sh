set -euo pipefail
root=/data1/shenglin.zeng/teacher_turning_20261026
cd "$root"
export LD_LIBRARY_PATH="$root/runtime/lib:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES=0 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 OMP_NUM_THREADS=4
timeout --signal=TERM --kill-after=5s 120s nice -n 10 env/bin/python -u optimizer_dtensor_probe.py
