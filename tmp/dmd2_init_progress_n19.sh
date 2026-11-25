ps -p 485570,485571,485572,485573 -o pid,stat,pcpu,rss,wchan:24
cat /sys/fs/cgroup/user.slice/memory.current /sys/fs/cgroup/user.slice/memory.events
cat /proc/485570/io
