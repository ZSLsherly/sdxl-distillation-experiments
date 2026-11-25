cd /data5/shenglin.zeng/sdxl_crossdistill
for name in gan_v2_preflight gan_v2_series gan_v2_assessment; do
  if test -f logs/$name.exit; then echo "$name EXIT $(cat logs/$name.exit)"; fi
  if test -f logs/$name.log; then tail -n 5 logs/$name.log; fi
done
for v in dmd gan4 joint4 joint34_matched; do
  if test -f logs/gan_v2_$v.log; then echo "TRIAL $v"; tail -n 2 logs/gan_v2_$v.log; fi
done
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
