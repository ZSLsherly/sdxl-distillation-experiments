set -eu
B=/data1/shenglin.zeng/dymd_prior_8_20261101
test ! -e "$B/controller.pid"
nohup python3 "$B/controller.py" > "$B/controller.log" 2>&1 < /dev/null &
echo $! > "$B/controller.pid"
cat "$B/controller.pid"
