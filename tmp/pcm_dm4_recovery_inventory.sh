set -eu
. /root/sdxl_research/env.sh
cd "$SDXL_ROOT/outputs/c_mix1_pcm_dm4_matched_time"
date -Iseconds
ls -la
find model/weights -maxdepth 2 -name complete.json -o -name cpu_validation.json
tail -8 train.log | cut -c1-160
ps -eo pid,ppid,comm,args | grep -E 'run_pcm_dm4|evaluate.py|train.py' | grep -v grep | cut -c1-260 || true
timeout 15 nvidia-smi --query-gpu=memory.free,utilization.gpu --format=csv || true
df -h .
