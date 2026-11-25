cd /data5/shenglin.zeng/sdxl_crossdistill
if test -f logs/gan_v3_preflight.exit; then cat logs/gan_v3_preflight.exit; fi
cat outputs/gan_stage_v3/preflight/status.json
tail -n 22 logs/gan_v3_preflight.log
