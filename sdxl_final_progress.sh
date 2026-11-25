source /data5/shenglin.zeng/env.sh
root=/data5/shenglin.zeng/sdxl_crossdistill
tail -n 4 "$root/logs/hparam_pipeline.log"
tail -n 4 "$root/logs/pilot_lr25e8_pcm1.log"
tail -n 5 "$root/logs/pilot_assessment.log"
