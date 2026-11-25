cd /data5/shenglin.zeng/sdxl_crossdistill
grep -B 18 'OutOfMemoryError' logs/parallel_mb32_adamcpu.log | head -n 25
if [ -f outputs/parallel_probe/mb16_adamcpu/benchmark.json ]; then cat outputs/parallel_probe/mb16_adamcpu/benchmark.json; fi
free -h
df -h /data5
