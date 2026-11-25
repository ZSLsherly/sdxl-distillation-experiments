source /data5/shenglin.zeng/env.sh
cd /data5/shenglin.zeng/sdxl_crossdistill
ps -p 3599028 --ppid 3599028 -o pid,ppid,stat,etime,pcpu,pmem,rss,args
free -h
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader
tail -n 16 logs/joint34_4gpu_20261019T195324.log
