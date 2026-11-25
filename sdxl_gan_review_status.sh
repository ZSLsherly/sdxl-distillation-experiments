cd /data5/shenglin.zeng/sdxl_crossdistill
cat logs/gan_calibration.exit
tail -n 8 logs/gan_calibration.log
cat outputs/gan_stage_pilot/calibration/summary.json
ps -eo pid,user,args | awk '$0 ~ /sdxl_gan_train[.]py/ && $0 !~ /awk/ {print}'
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
