set -eu
. /root/sdxl_research/env.sh
date -Iseconds
df -h /root /root/autodl-tmp
du -h -d 2 /root/autodl-tmp/sdxl_runs 2>/dev/null | sort -h | tail -25
ps -eo pid,ppid,comm,args | awk '$3 ~ /^python/ && /run_|train.py|pull.py|finish/ {print substr($0,1,220)}'
for n in A_pcm10_mix1_g160 C_mix1_pcm_dm4_fake4_low30_g160; do
 echo "$n"
 tail -12 "$SDXL_ROOT/outputs/$n/queue.log"
 test ! -f "$SDXL_ROOT/outputs/$n/model/status.json" || cat "$SDXL_ROOT/outputs/$n/model/status.json"
done
