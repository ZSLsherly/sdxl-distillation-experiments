source /data5/shenglin.zeng/env.sh
root=/data5/shenglin.zeng/sdxl_crossdistill
tail -n 5 "$root/logs/hparam_pipeline.log"
tail -n 6 "$root/logs/teacher_turning.log"
tail -n 3 "$root/logs/pilot_lr5e7_pcm1.log"
if [ -f "$root/logs/pilot_lr25e8_pcm1.log" ]; then tail -n 4 "$root/logs/pilot_lr25e8_pcm1.log";fi
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
