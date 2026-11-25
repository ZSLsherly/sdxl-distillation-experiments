cd /data5/shenglin.zeng/sdxl_crossdistill
if test -f logs/gan_v3_preflight.exit; then cat logs/gan_v3_preflight.exit; fi
tail -n 18 logs/gan_v3_preflight.log
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
