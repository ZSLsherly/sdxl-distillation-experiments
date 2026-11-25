cd /data5/shenglin.zeng/a50k_20261124
date
tail -3 logs/singlepass_queue.log
tail -6 logs/A50K_singlepass.log 2>/dev/null || true
cat outputs/A50K_singlepass/status.json 2>/dev/null || true
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
