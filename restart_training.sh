set -eu
source /data5/shenglin.zeng/env.sh
cd "$CROSSDISTILL_ROOT"
stamp=$(date +%Y%m%d_%H%M%S)
if [ -f logs/train_smoke.log ]; then mv logs/train_smoke.log "logs/train_smoke_before_${stamp}.log"; fi
if [ -f logs/smoke_pipeline.exit ]; then mv logs/smoke_pipeline.exit "logs/smoke_pipeline_before_${stamp}.exit"; fi
nohup bash -c 'source /data5/shenglin.zeng/env.sh; cd "$CROSSDISTILL_ROOT"; nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu,power.draw --format=csv -l 1 > logs/training_gpu_samples.csv & monitor=$!; trap '\''code=$?; kill "$monitor" 2>/dev/null || true; echo "$code" > logs/smoke_pipeline.exit'\'' EXIT; torchrun --standalone --nproc_per_node=4 -m scripts.train --config=smoke_config.py > logs/train_smoke.log 2>&1' > logs/training_restart.log 2>&1 < /dev/null &
printf 'TRAIN_RESTART_PID=%s\n' "$!"
