source /data5/shenglin.zeng/env.sh
root="$CROSSDISTILL_ROOT/sdxl_crossdistill"
tail -n 150 "$root/logs/finish.log"
ps -eo pid,user,args | grep -E '[s]dxl_compare.py' | head -n 8
