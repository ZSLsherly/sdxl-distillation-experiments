source /data5/shenglin.zeng/env.sh
root="$CROSSDISTILL_ROOT/sdxl_crossdistill"
tail -n 12 "$root/logs/download.log"
du -sh "$root/models"
