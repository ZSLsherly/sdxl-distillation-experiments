set -eu
. /root/sdxl_research/env.sh
p="$SDXL_ROOT/outputs/bc_winner_late_mix_20261102"
tail -6 "$SDXL_WORK/logs/bc_coco_winner_queue.log"
if test -e "$p/coco10k/status_0.json"; then cat "$p/coco10k/status_0.json"; fi
if test -e "$p/coco10k/worker.log"; then tail -3 "$p/coco10k/worker.log"; fi
for f in scores.json complete.json; do if test -e "$p/coco10k/$f"; then cat "$p/coco10k/$f"; fi; done
if test -e "$p/selection.json"; then cat "$p/selection.json"; fi
for f in status_0.json scores.json complete.json; do
    if test -e "$p/coco10k_A/$f"; then echo "A $f"; cat "$p/coco10k_A/$f"; fi
done
