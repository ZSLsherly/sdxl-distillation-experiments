set -eu
r=/data1/shenglin.zeng/coco_missing_20261111
p=/data1/shenglin.zeng/teacher_turning_20261026/env/bin/python
cd "$r"
exec 9>launch.lock
flock -n 9
if test -f queue.pid && kill -0 "$(cat queue.pid)" 2>/dev/null; then echo ALREADY_ACTIVE; cat queue.pid; exit; fi
nvidia-smi -i 4 --query-gpu=index,uuid,memory.free,utilization.gpu --format=csv,noheader,nounits
nvidia-smi -i 4 --query-compute-apps=pid --format=csv,noheader,nounits
nohup env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 "$p" -u code/run_missing.py 9>&- >>queue.log 2>&1 </dev/null &
echo $! >queue.pid
sleep 3
kill -0 "$(cat queue.pid)"
cat queue.pid
tail -8 queue.log
