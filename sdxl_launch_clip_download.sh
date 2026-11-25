source /data5/shenglin.zeng/env.sh
root=/data5/shenglin.zeng/sdxl_crossdistill
cd "$root"
nohup python src/sdxl_download_clip.py > logs/clip_download.log 2>&1 < /dev/null &
echo $! > logs/clip_download.pid
cat logs/clip_download.pid
