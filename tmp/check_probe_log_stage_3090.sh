date -Is
root=/data1/shenglin.zeng/teacher_turning_20261026/outputs/turning_boundary_ablation_20261027/short_normalized_u_retry1
ls -l --time-style=long-iso "$root/config.json" "$root/rank0.jsonl" "$root/status.json" 2>/dev/null || true
if test -f "$root/rank0.jsonl";then tail -4 "$root/rank0.jsonl";fi
nproc
