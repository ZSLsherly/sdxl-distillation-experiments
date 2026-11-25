root=/data1/shenglin.zeng/teacher_turning_20261026
cat "$root/tmp/side_teacher_guard_20261029.log"
for f in "$root/outputs/side_teacher_perf_20261029/"*.log; do echo "$f"; tail -n 12 "$f"; done
cat "$root/outputs/side_teacher_perf_20261029/complete.json" 2>/dev/null || true
nvidia-smi --id=1,4 --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
