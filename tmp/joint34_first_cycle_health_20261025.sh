set -e
nvidia-smi --query-gpu=index,memory.used,utilization.gpu,power.draw --format=csv,noheader
free -h
ps -eo pid,ppid,stat,pcpu,rss,comm | awk '$2==2658520 || $1==2658520 || $1==2658510'
tail -n 6 /data1/shenglin.zeng/sdxl_eval_uniform_20261025/outputs/joint34_uniform_10k_2gpu_20261025/phase1_0.log
