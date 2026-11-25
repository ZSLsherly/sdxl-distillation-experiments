date -Is
ps -p 30181 -o pid,etime,stat,pcpu,pmem,rss,args
cat /proc/30181/wchan 2>/dev/null || true
tail -12 /data1/shenglin.zeng/teacher_turning_20261026/stage_smoke/worker0.log
