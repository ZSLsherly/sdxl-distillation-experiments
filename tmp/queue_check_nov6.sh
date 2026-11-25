set -eu
. /root/sdxl_research/env.sh
date -Iseconds
ps -eo pid,ppid,comm,args | awk '$3 ~ /^python/ && /run_|train.py|evaluate.py|score.py/ {print substr($0,1,220)}'
tail -5 "$SDXL_ROOT/outputs/A_pcm10_mix1_g160/queue.log"
grep -nE 'evaluate_checkpoint|comparison.json|COMPLETE' "$SDXL_ROOT/tmp/A_pcm10_mix1_g160/code/run_A10_mix1.py"
