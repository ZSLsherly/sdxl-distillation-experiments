ROOT=/data5/shenglin.zeng
ps -u "$(id -un)" -o pid,ppid,stat,etime,wchan:25,args | grep -E 'scripts.train|2987800|training_gpu_samples' | head -12
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
tail -5 "$ROOT/logs/training_restart.log"
