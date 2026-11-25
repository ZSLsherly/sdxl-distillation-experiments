set -eu
. /root/sdxl_research/env.sh
d="$SDXL_ROOT/outputs/interval_D_20261103/D";p="$SDXL_ROOT/outputs/bc_winner_late_mix_20261102/coco10k_C_mix1"
date '+%F %T %z'
cat "$d/status.json"; echo
tail -7 "$p/queue.log"; test ! -f "$p/worker.log" || tail -1 "$p/worker.log"
test ! -f "$p/status_0.json" || cat "$p/status_0.json"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
