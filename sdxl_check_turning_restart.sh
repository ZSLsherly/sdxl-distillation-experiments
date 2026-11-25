source /data5/shenglin.zeng/env.sh
root=/data5/shenglin.zeng/sdxl_crossdistill
mv "$root/logs/teacher_turning.log" "$root/logs/teacher_turning_first_attempt.log"
tail -n 7 "$root/logs/pilot_lr5e7_pcm1.log"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
