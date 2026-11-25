ROOT=/data5/shenglin.zeng
tail -10 "$ROOT/logs/gpu_comm.log"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
tail -8 "$ROOT/logs/install_deps.log"
tail -4 "$ROOT/logs/install_torch.log"
