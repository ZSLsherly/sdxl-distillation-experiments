set -eu
ROOT=/data5/shenglin.zeng
printf 'BENCHMARK PROCESSES\n'
ps -u "$(id -un)" -o pid,args | grep 'gpu_comm_bench.py' | head -8
if [ -r /proc/2976250/cmdline ] && tr '\0' ' ' < /proc/2976250/cmdline | grep -q '/data5/shenglin.zeng/benchmarks/gpu_comm_bench.py'; then kill -TERM 2976250; fi
sleep 3
export TMPDIR="$ROOT/tmp" XDG_CACHE_HOME="$ROOT/cache" TORCH_HOME="$ROOT/cache/torch" TRITON_CACHE_DIR="$ROOT/cache/triton" PYTHONDONTWRITEBYTECODE=1
export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 NCCL_DEBUG=INFO
timeout 150 "$ROOT/envs/crossdistill/bin/torchrun" --standalone --nproc_per_node=4 "$ROOT/benchmarks/gpu_comm_bench.py" > "$ROOT/logs/gpu_comm_host.log" 2>&1
tail -12 "$ROOT/logs/gpu_comm_host.log"
