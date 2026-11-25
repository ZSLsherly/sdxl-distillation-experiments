set -eu
out=/data1/shenglin.zeng/teacher_turning_20261026/outputs/turning_boundary_ablation_20261027
date -Is
tail -5 "$out/shared_probe.log" 2>/dev/null || true
for name in normalized_u raw_angle;do
 if test -f "$out/short_$name.log";then printf '\n%s\n' "$name";tail -5 "$out/short_$name.log";fi
 if test -f "$out/short_$name/benchmark.json";then cat "$out/short_$name/benchmark.json";fi
done
for name in paired_GPU_probe_complete.json paired_GPU_probe_error.json;do if test -f "$out/$name";then cat "$out/$name";fi;done
nvidia-smi -i 0 --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits
free -h
