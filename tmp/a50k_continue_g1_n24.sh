set -eu
cd /data5/shenglin.zeng/a50k_20261124
export TMPDIR="$PWD/cache/TMPDIR" OMP_NUM_THREADS=4
nohup /data5/shenglin.zeng/envs/crossdistill/bin/python -u code/queue50.py >> logs/queue.log 2>&1 </dev/null &
echo "CONTINUATION_PID=$!"
sleep 2
tail -4 logs/queue.log
