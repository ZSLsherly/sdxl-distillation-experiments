set -eu
out=/data1/shenglin.zeng/teacher_turning_20261026/outputs/turning_boundary_ablation_20261027
date -Is
tail -4 "$out/shared_probe_retry1.log" 2>/dev/null || true
for name in normalized_u raw_angle;do
 if test -f "$out/short_${name}_retry1.log";then printf '\n%s\n' "$name";tail -5 "$out/short_${name}_retry1.log";fi
 if test -f "$out/short_${name}_retry1/benchmark.json";then cat "$out/short_${name}_retry1/benchmark.json";fi
done
for name in paired_GPU_probe_retry1_complete.json paired_GPU_probe_retry1_error.json;do if test -f "$out/$name";then cat "$out/$name";fi;done
nvidia-smi -i 0 --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits
free -h
