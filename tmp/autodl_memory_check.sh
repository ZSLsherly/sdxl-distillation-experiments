set -eu
root=/root/autodl-tmp/sdxl_research
for f in /sys/fs/cgroup/memory.max /sys/fs/cgroup/memory.events /sys/fs/cgroup/memory.current /sys/fs/cgroup/memory/memory.limit_in_bytes /sys/fs/cgroup/memory/memory.failcnt; do
 if test -f "$f"; then echo "$f";cat "$f";fi
done
cat "$root/env.sh"
ls -lh "$root/datasets/parallel_10k_seed20261018"
find "$root/models/sdxl-base-1.0" -maxdepth 2 -type f -printf '%P %s bytes\n'
