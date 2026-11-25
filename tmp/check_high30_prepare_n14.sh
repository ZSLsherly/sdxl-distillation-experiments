ps -eo pid,etime,pcpu,pmem,args | grep '[p]repare.py'
ls -lh /data1/shenglin.zeng/high30_parallel_eval_20261114/checkpoint
test ! -e /data1/shenglin.zeng/high30_parallel_eval_20261114/assets_ready.json || cat /data1/shenglin.zeng/high30_parallel_eval_20261114/assets_ready.json
