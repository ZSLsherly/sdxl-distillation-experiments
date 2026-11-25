R=/data1/shenglin.zeng/dmd2_matched_10k_20261119
for f in "$R"/runs/smoke_micro16/rank*.jsonl; do echo "$f"; tail -n 2 "$f"; done
cat /sys/fs/cgroup/user.slice/memory.current
