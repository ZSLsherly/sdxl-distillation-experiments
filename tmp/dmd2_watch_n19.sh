R=/data1/shenglin.zeng/dmd2_matched_10k_20261119
date
tail -n 8 "$R/logs/queue.log"
tail -n 40 "$R/logs/smoke_micro16.log"
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader
