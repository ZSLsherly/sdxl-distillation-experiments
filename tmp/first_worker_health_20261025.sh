set -e
ps -p 2509374,2509383,2509389,2509396 -o pid,etime,stat,pcpu,rss,wchan:25
nvidia-smi --query-gpu=index,memory.used,utilization.gpu,power.draw --format=csv,noheader
ls -l /data1/shenglin.zeng/sdxl_eval_uniform_20261025/outputs/coco10k/features/joint4_uniform
