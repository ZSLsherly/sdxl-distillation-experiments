set -e
for p in /sys/fs/cgroup/memory /sys/fs/cgroup/memory/user.slice /sys/fs/cgroup/memory/user.slice/user-1017.slice; do
 printf '%s\n' "$p"
 for f in memory.limit_in_bytes memory.soft_limit_in_bytes memory.usage_in_bytes memory.failcnt; do
  if test -f "$p/$f"; then printf '%s ' "$f"; cat "$p/$f"; fi
 done
done
