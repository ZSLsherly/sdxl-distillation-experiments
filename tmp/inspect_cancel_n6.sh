set -eu
. /root/sdxl_research/env.sh
date -Iseconds
ps -eo pid,ppid,comm,args | awk '$3 ~ /^python/ && /run_|train.py|evaluate.py|score.py|pull.py/ {print substr($0,1,240)}'
cd "$SDXL_ROOT/tmp/A_pcm10_mix1_g160/code"
grep -nE 'STOP_REQUEST|def run_gpu|start_new_session|signal|logfile.name' run_dm1_uniform_control.py train.py
cat "$SDXL_ROOT/outputs/A_pcm10_mix1_g160/model/status.json"
