set -eu
B=/data1/shenglin.zeng/dymd_prior_8_20261101
pid=$(cat "$B/controller.pid")
case "$(ps -p "$pid" -o args=)" in *dymd_prior_8_20261101/controller.py*) kill -TERM "$pid";; *) exit 2;; esac
nohup python3 "$B/controller.py" >> "$B/controller.log" 2>&1 < /dev/null &
echo $! > "$B/controller.pid"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
