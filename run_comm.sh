set -eu
ROOT=/data5/shenglin.zeng
export TMPDIR="$ROOT/tmp" XDG_CACHE_HOME="$ROOT/cache" TORCH_HOME="$ROOT/cache/torch" TRITON_CACHE_DIR="$ROOT/cache/triton" PYTHONDONTWRITEBYTECODE=1
"$ROOT/envs/crossdistill/bin/python" -c 'import torch;print(torch.__version__,torch.version.cuda,torch.cuda.is_available(),torch.cuda.device_count())'
"$ROOT/envs/crossdistill/bin/torchrun" --standalone --nproc_per_node=4 "$ROOT/benchmarks/gpu_comm_bench.py" 2>&1 | tee "$ROOT/logs/gpu_comm.log"
