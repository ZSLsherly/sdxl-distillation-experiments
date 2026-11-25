ps -p 2140773,2141440,2141441,2141445,2141447 -o pid,stat,args
nvidia-smi --id=0,1,5,6 --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
R=/data1/shenglin.zeng/dmd2_matched_10k_20261119/coco10k_official_20261121
tail -n 5 "$R/logs/queue.log"
