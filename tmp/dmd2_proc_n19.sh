ps -p 455096,455222,455398,455399,455400,455401 -o pid,stat,etime,pcpu,pmem,rss,wchan:24,args
free -h
nvidia-smi -i 0 --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader
