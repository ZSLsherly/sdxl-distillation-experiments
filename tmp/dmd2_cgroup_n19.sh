mount | grep cgroup
find /sys/fs/cgroup -path '*user-1009.slice*' -name '*memory*' 2>/dev/null | head -n 20
cat /sys/fs/cgroup/unified/user.slice/user-1009.slice/memory.max /sys/fs/cgroup/unified/user.slice/user-1009.slice/memory.current /sys/fs/cgroup/unified/user.slice/user-1009.slice/memory.events 2>/dev/null || true
ps -p 455398,455399,455400,455401 -o pid,stat,pcpu,rss
