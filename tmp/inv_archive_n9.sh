df -h /root /root/autodl-tmp
du -xh --max-depth=1 /root | sort -h | tail -20
du -h --max-depth=1 /root/autodl-tmp/sdxl_runs/outputs | sort -h | tail -30
find /root -name student_rank0.pt -printf '%s %p\n' 2>/dev/null
ps -eo pid,comm,args | awk '$2 ~ /python|torchrun/ && /train.py|run_terminal|queue/ {print substr($0,1,180)}'
