set -eu
. /root/sdxl_research/env.sh
date -Iseconds
df -h /root/autodl-tmp
nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader,nounits
ps -eo pid,ppid,args | grep -E 'run_.*n1[34].py|torch.distributed.run' | grep -v grep || true
for n in A_clean_global_g160_20261113 A_pcm1_terminal_dm_g160_20261113; do
 r="$SDXL_ROOT/outputs/$n"
 echo "$n"
 tail -8 "$r/queue.log"
 test ! -f "$r/model/status.json" || cat "$r/model/status.json"
 test ! -f "$r/complete.json" || cat "$r/complete.json"
done
