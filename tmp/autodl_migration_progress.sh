set -eu
root=/root/autodl-tmp/sdxl_research
ps -eo pid,ppid,etime,pcpu,rss,stat,args | grep -E '[r]sync|[m]igrate_system.sh'
cat /sys/fs/cgroup/cpu.max
ls -ld /root/sdxl_research/*
tail -n 12 "$root/logs/system_migration.log"
