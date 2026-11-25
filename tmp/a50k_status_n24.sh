R=/data/shenglin.zeng/a50k_20261123
if test -d "$R"; then
 cat "$R/records/prepare_status.json"
 tail -8 "$R/records/build.log" 2>/dev/null
 ls -lh "$R/datasets/sources" "$R/datasets/parallel_50k_seed20261018"
 cat "$R/datasets/parallel_50k_seed20261018/validation.json" 2>/dev/null
fi
ps -u "$(id -un)" -o pid=,comm=,args= | awk '/[t]orchrun|[t]rain.py|[t]rain_matched.py|[p]repare_queue.py|[b]uild_50k.py/ && !/awk/ {print $1,$2,substr($0,1,170)}'
