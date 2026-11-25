set -e
cd /data1/shenglin.zeng/sdxl_eval_uniform_20261025
ls -ld env/lib/python3.12/site-packages/torch models checkpoints outputs 2>/dev/null || true
ps -C tar -o pid,etime,pcpu,comm
