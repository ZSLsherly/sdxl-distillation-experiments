find /data3/shenglin.zeng/clip_missing_20261113/models/clip -maxdepth 1 -type f -printf '%f %s %TY-%Tm-%Td %TH:%TM:%TS\n' 2>/dev/null
ls /data3/shenglin.zeng/clip_missing_20261113/validation_images 2>/dev/null | wc -l
tail -4 /data3/shenglin.zeng/clip_missing_20261113/queue_5.log
