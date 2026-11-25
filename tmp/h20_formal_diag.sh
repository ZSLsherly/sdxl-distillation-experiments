cd /root/autodl-tmp/h20_C_mix1_formal_20261103
pstree -ap 11397 | head -25
ps -eo pid,ppid,etime,stat,pcpu,rss,wchan:24,args | awk '$2==11397 || $2==11402' | cut -c1-210
p=$(pgrep -P 11397 | head -1)
if [ -n "$p" ];then ls -l /proc/$p/fd | tail -14;cat /proc/$p/io;fi
tail -20 queue.log
