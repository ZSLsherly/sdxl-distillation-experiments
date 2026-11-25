cat /proc/455398/cgroup
cat /sys/fs/cgroup/memory.max /sys/fs/cgroup/memory.current /sys/fs/cgroup/memory.events 2>/dev/null || true
tail -n 12 /data1/shenglin.zeng/dmd2_matched_10k_20261119/logs/smoke_micro16.log
