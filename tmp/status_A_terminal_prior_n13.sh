set -eu
. /root/sdxl_research/env.sh
r="$SDXL_ROOT/outputs/A_pcm10_terminal_low30_g160_20261113"
date -Iseconds
tail -8 "$r/queue.log"
test ! -f "$r/model/status.json" || cat "$r/model/status.json"
test ! -f "$r/train.log" || tail -5 "$r/train.log"
nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader,nounits
