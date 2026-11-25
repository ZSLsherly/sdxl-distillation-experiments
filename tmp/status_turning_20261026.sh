set -eu
root=/data1/shenglin.zeng/teacher_turning_20261026
date -Is
if test -d "$root/results"; then
 find "$root/results" -name '*.json' | wc -l
 for mode in base_dt10_bf16 fine_dt5_bf16 precision_dt10_fp32; do printf '%s ' "$mode";find "$root/results" -name "$mode*.json" | wc -l;done
fi
for log in guard.log worker0.log worker1.log;do
 if test -f "$root/$log";then printf '%s\n' "$log";tail -4 "$root/$log";fi
done
for record in complete.json error.json;do if test -f "$root/$record";then cat "$root/$record";fi;done
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
