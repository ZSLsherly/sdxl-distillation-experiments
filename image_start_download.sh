set -eu
source /data5/shenglin.zeng/env.sh
cd "$CROSSDISTILL_ROOT/image_crossdistill"
nohup python -u src/image_download_model.py > logs/download.log 2>&1 < /dev/null &
echo $! > logs/download.pid
cat logs/download.pid
