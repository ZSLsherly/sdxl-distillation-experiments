set -eu
source /data5/shenglin.zeng/env.sh
cd "$CROSSDISTILL_ROOT/sdxl_crossdistill"
export HTTPS_PROXY=http://127.0.0.1:18797 HTTP_PROXY=http://127.0.0.1:18797
export HF_ENDPOINT=https://huggingface.co
nohup python -u src/sdxl_download_model.py > logs/download.log 2>&1 < /dev/null &
echo $! > logs/download.pid
cat logs/download.pid
