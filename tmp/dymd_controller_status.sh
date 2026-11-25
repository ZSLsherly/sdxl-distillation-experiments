B=/data1/shenglin.zeng/dymd_prior_8_20261101
cat "$B/status.json"
tail -12 "$B/install_fast.log"
ps -p "$(cat "$B/controller.pid")" -o pid,etime,args
