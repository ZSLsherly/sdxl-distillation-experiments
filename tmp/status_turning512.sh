set -eu
root=/data1/shenglin.zeng/teacher_turning_20261026/results512
date -Is
test ! -f "$root/status.json" || cat "$root/status.json"
for log in guard.log worker0.log worker1.log worker2.log;do
 if test -f "$root/$log";then printf '\n%s\n' "$log";tail -4 "$root/$log";fi
done
for f in complete.json error.json;do if test -f "$root/$f";then cat "$root/$f";fi;done
