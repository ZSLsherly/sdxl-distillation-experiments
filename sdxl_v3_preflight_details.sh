cd /data5/shenglin.zeng/sdxl_crossdistill
if test -f outputs/gan_stage_v3/preflight/status.json; then cat outputs/gan_stage_v3/preflight/status.json; fi
if test -f outputs/gan_v3_calibration.json; then cat outputs/gan_v3_calibration.json; fi
tail -n 18 logs/gan_v3_preflight.log
