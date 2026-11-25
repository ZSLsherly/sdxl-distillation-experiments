set -eu
date -Is
ps -eo pid,ppid,stat,pcpu,pmem,rss,etime,wchan:28,args | awk '$2==19090 || $1==19090 || $1==19079 {print}'
for p in 19110 19111;do
 if test -e /proc/$p/io;then echo PID=$p;cat /proc/$p/io;fi
done
free -h
df -h /dev/shm
