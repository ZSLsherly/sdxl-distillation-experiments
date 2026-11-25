set -eu
out=/data5/shenglin.zeng/sdxl_crossdistill/outputs/boundary_dmd_10k_a100_20261025_v1
date -Is
tail -4 "$out/workflow.jsonl" 2>/dev/null || true
for phase in short_retry1 train; do
  if test -f "$out/$phase.log"; then printf '\n%s\n' "$phase"; tail -12 "$out/$phase.log"; fi
  if test -f "$out/$phase/status.json"; then cat "$out/$phase/status.json"; fi
done
nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits
free -h
