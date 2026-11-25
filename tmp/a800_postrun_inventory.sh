set -eu
. /root/sdxl_research/env.sh
date -Iseconds
ps -eo pid,ppid,comm,args | awk '$3 ~ /^python/ && /queue|after_|run_pcm_dm4|evaluate.py/ {print substr($0,1,240)}'
f="$SDXL_ROOT/tmp/c_mix1_pcm_dm4_matched_time/code/run_pcm_dm4_comparison.py"
grep -nE 'for g,ck in|scores=evaluate_checkpoint|save\(job./.complete.json' "$f"
