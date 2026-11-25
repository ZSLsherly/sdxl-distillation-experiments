source /data5/shenglin.zeng/env.sh
root=/data5/shenglin.zeng/sdxl_crossdistill
tail -n 9 "$root/logs/teacher_stages.log"
tail -n 3 "$root/logs/hparam_pipeline.log"
if [ -f "$root/logs/pilot_lr5e7_pcm10.log" ];then tail -n 7 "$root/logs/pilot_lr5e7_pcm10.log";fi
