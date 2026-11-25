B=/data1/shenglin.zeng/dymd_prior_8_20261101
date -u
cat "$B/status.json"
ps -p "$(cat "$B/controller.pid")" -o pid,stat,etime,args
test ! -f "$B/progress.json" || cat "$B/progress.json"
tail -5 "$B/controller.log"
test ! -f "$B/run.log" || tail -15 "$B/run.log"
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits
df -h "$B"
find "$B/profiles" -maxdepth 1 -name '*.complete' 2>/dev/null | wc -l
