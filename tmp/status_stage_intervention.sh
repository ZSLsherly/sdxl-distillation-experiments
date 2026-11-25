set -eu
root=/data1/shenglin.zeng/teacher_turning_20261026
date -Is
for name in stage_smoke stage_intervention;do
 if test -d "$root/$name";then
  printf '\n%s\n' "$name"
  for f in status.json complete.json error.json;do if test -f "$root/$name/$f";then cat "$root/$name/$f";fi;done
  for f in "$root/$name"/worker*.log;do if test -f "$f";then tail -6 "$f";fi;done
 fi
done
