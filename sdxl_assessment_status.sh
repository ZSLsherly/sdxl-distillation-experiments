source /data5/shenglin.zeng/env.sh
root=/data5/shenglin.zeng/sdxl_crossdistill
tail -n 10 "$root/logs/pilot_assessment.log"
if [ -f "$root/logs/pilot_assessment.exit" ];then cat "$root/logs/pilot_assessment.exit";fi
