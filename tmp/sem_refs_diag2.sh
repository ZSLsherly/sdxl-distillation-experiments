cd /data1/shenglin.zeng/sdxl_eval_uniform_20261025/outputs/semantic_references_1024
uptime -s; free -h;df -h .
ls -lh models/baselines/dmd2/
tail -8 xfer/transfer.log
ls reference/sdxl | wc -l;ls reference/dmd2 | wc -l
nvidia-smi --query-gpu=index,uuid,memory.free,utilization.gpu,temperature.gpu --format=csv,noheader,nounits
cat resume.log | tail -8
