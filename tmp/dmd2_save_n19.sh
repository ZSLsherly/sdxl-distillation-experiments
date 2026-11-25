R=/data1/shenglin.zeng/dmd2_matched_10k_20261119
find "$R/runs/smoke_micro16/eval/g000002" -maxdepth 1 -type f -printf '%f\n' 2>/dev/null
ls -lh "$R/runs/smoke_micro16/student.pt" 2>/dev/null || true
tail -n 4 "$R/logs/smoke_micro16.log"
cat /sys/fs/cgroup/user.slice/memory.current
