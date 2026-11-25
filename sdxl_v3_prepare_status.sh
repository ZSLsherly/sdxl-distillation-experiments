set -e
cd /data5/shenglin.zeng/sdxl_crossdistill
tail -n 5 logs/gan_v3_prepare.log
ls -lh datasets/gan_v3
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
