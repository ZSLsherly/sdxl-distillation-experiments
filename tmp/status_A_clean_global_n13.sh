set -eu
. /root/sdxl_research/env.sh
date -Iseconds
for n in A_pcm10_terminal_low30_g160_20261113 A_clean_global_g160_20261113; do
 r="$SDXL_ROOT/outputs/$n"
 echo "$n"
 tail -6 "$r/queue.log"
 if test -f "$r/model/status.json"; then cat "$r/model/status.json"; fi
 if test -f "$r/first_update_validation.json"; then cat "$r/first_update_validation.json"; fi
 if test -f "$r/complete.json"; then cat "$r/complete.json"; fi
done
ps -eo pid,ppid,args | grep -E 'run_A_.*n13.py|torch.distributed.run' | grep -v grep
nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader,nounits
