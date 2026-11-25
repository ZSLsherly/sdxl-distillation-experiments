date -Is
out=/data1/shenglin.zeng/teacher_turning_20261026/outputs/turning_boundary_ablation_20261027
cat "$out/short_normalized_u_retry1/status.json"
wc -l "$out/shared_probe_retry1.log"
tail -15 "$out/shared_probe_retry1.log"
if test -f "$out/YIELD_RETRY1";then echo 'PAUSE_PRESENT';else echo 'NO_PAUSE';fi
