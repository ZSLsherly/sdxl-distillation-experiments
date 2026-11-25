set -e
cd /data5/shenglin.zeng/sdxl_crossdistill
for i in 113 118; do
  echo CASE_$i
  tail -n 4 logs/original_control_${i}.log
done
