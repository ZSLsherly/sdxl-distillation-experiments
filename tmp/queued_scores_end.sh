set -eu
. /root/sdxl_research/env.sh
d="$SDXL_ROOT/outputs/bc_winner_late_mix_20261102"
for n in 1 2 3 4 5 6; do
  if test -f "$d/coco10k/complete.json"; then cat "$d/coco10k/complete.json"; break; fi
  sleep 4
done
tail -4 "$d/coco10k/metrics.log"
if test -e "$d/coco10k_A/worker.log"; then tail -1 "$d/coco10k_A/worker.log"; fi
