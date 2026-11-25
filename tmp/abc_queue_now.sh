set -eu
. /root/sdxl_research/env.sh
p="$SDXL_ROOT/outputs/bc_winner_late_mix_20261102"
date '+%F %T %z'
ps -eo pid,comm,args | awk '($2 ~ /python/ && /run_bc_coco_then_winner.py|evaluate.py worker|evaluate.py metrics|train.py/)'
tail -5 "$SDXL_WORK/logs/bc_coco_winner_queue.log"
grep 'METHOD_DONE' "$p/coco10k/worker.log" | tail -3 || true
for f in coco10k/status_0.json coco10k/scores.json coco10k/complete.json coco10k_A/status_0.json coco10k_A/scores.json coco10k_A/complete.json selection.json late_mix/status.json training_complete.json; do if test -f "$p/$f"; then echo "$f"; head -c 1000 "$p/$f"; echo; fi; done
