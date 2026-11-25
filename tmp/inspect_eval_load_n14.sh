date
free -h
ps -p 31509,31511,31512,31517,31531,31538 -o pid,stat,etime,pcpu,rss,wchan:24
for p in 31509 31511 31512 31517 31531 31538; do echo PID=$p; cat /proc/$p/io | head -n 4; done
