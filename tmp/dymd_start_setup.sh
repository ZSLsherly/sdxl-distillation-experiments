set -eu
B=/data1/shenglin.zeng/dymd_prior_8_20261101
nohup bash "$B/install.sh" > "$B/install.log" 2>&1 < /dev/null &
echo "INSTALL_PID=$!"
nohup bash "$B/download.sh" > "$B/download.log" 2>&1 < /dev/null &
echo "DOWNLOAD_PID=$!"
