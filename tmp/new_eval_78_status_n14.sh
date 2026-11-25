cd /data1/shenglin.zeng/new_models_eval_20261114
cat queue.log 2>/dev/null
test ! -f queue.pid || ps -p "$(cat queue.pid)" -o pid,etime,args
find results -maxdepth 2 -name ready.json -o -name all_complete.json -o -name status_0.json
tail -n 5 results/*/worker.log 2>/dev/null
nvidia-smi -i 4 --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv,noheader
