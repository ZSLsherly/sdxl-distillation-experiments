set -eu
source /data5/shenglin.zeng/env.sh
root="$CROSSDISTILL_ROOT/sdxl_crossdistill"
cd "$root"
export HTTPS_PROXY=http://127.0.0.1:18797 HTTP_PROXY=http://127.0.0.1:18797 HF_ENDPOINT=https://huggingface.co
nohup python -u src/sdxl_download_baselines.py > logs/baselines_download.log 2>&1 < /dev/null &
echo $! > logs/baselines_download.pid
cat logs/baselines_download.pid
