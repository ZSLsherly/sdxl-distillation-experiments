source /data5/shenglin.zeng/env.sh
root=/data5/shenglin.zeng/sdxl_crossdistill
cd "$root"
nohup python src/sdxl_stage_metrics.py > logs/stage_metrics.log 2>&1 < /dev/null &
echo $! > logs/stage_metrics.pid
cat logs/stage_metrics.pid
