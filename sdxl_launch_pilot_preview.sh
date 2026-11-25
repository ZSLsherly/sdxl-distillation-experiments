source /data5/shenglin.zeng/env.sh
cd /data5/shenglin.zeng/sdxl_crossdistill
nohup python src/sdxl_preview_pilot.py > logs/pilot_preview.log 2>&1 < /dev/null &
echo $! > logs/pilot_preview.pid
cat logs/pilot_preview.pid
