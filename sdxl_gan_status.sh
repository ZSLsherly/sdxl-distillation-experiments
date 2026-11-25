source /data5/shenglin.zeng/env.sh
cd /data5/shenglin.zeng/sdxl_crossdistill
for name in gan_calibration gan_stage_pilot; do
  if test -f logs/$name.exit; then echo "$name EXIT $(cat logs/$name.exit)"; fi
  if test -f logs/$name.log; then tail -n 6 logs/$name.log; fi
done
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
