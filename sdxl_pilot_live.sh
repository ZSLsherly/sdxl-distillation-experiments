source /data5/shenglin.zeng/env.sh
root=/data5/shenglin.zeng/sdxl_crossdistill
tail -n 3 "$root/logs/hparam_pipeline.log"
for f in "$root"/logs/pilot_lr*.log;do echo "$(basename "$f")";tail -n 2 "$f";done
tail -n 3 "$root/logs/clip_download.log"
ls -lh "$root/models/metrics"
