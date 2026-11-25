set -eu
B=/data1/shenglin.zeng/dymd_prior_8_20261101
test "$(ps -p 2746447 -o ppid= | tr -d ' ')" = 2746445
kill -TERM 2746447
export http_proxy=http://127.0.0.1:18897 https_proxy=http://127.0.0.1:18897 HF_HUB_DISABLE_XET=1
nohup /data1/shenglin.zeng/miniconda3/envs/sherly/bin/python "$B/download_parallel.py" > "$B/download_parallel.log" 2>&1 < /dev/null &
echo PARALLEL_DOWNLOAD_PID=$!
tail -4 "$B/install.log"
