cd /data5/shenglin.zeng/sdxl_crossdistill
if test -f logs/gan_stage_pilot.exit; then echo "SERIES_EXIT $(cat logs/gan_stage_pilot.exit)"; fi
if test -f logs/gan_stage_pilot.log; then tail -n 10 logs/gan_stage_pilot.log; fi
for variant in preflight dmd gan4 joint4 joint34; do
  if test -f logs/gan_${variant}.log; then echo "ACTIVE_OR_FINISHED $variant"; tail -n 4 logs/gan_${variant}.log; fi
done
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
