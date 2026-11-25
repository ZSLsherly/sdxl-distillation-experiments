root=/root/autodl-tmp/sdxl_research
ps -eo pid,etime,rss,stat,args | grep '[v]alidate_lowmem.py'
tail -n 12 "$root/logs/validate_lowmem.log"
cat /sys/fs/cgroup/memory.events
