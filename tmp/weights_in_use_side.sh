set -eu
ps -eo pid,ppid,comm,args | awk '$3 ~ /python/ && /train.py|evaluate.py|run_.*py|clip_four/ {print substr($0,1,520)}'
find /root/autodl-tmp/sdxl_runs/outputs -maxdepth 3 -name config.json -mmin -60 -print | tail -8
