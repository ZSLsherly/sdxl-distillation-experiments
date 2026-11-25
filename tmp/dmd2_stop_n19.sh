R=/data1/shenglin.zeng/dmd2_matched_10k_20261119
touch "$R/STOP_REQUEST"
sleep 8
ps -p 455096,455222,455398,455399,455400,455401 -o pid,stat,args
tail -n 8 "$R/logs/queue.log"
