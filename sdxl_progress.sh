source /data5/shenglin.zeng/env.sh
root="$CROSSDISTILL_ROOT/sdxl_crossdistill"
tail -n 5 "$root/logs/download.log"
tail -n 5 "$root/logs/data_download.log"
tail -n 6 "$root/logs/prepare.log"
tail -n 8 "$root/logs/train.log"
du -sh "$root/models" "$root/datasets"
if [ -f "$root/logs/train.exit" ]; then cat "$root/logs/train.exit"; fi
