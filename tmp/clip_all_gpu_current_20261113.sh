nvidia-smi -i 5 --query-compute-apps=pid,used_memory --format=csv,noheader
ps -ef | grep '[c]lip_missing_20261113\|[w]orker.py\|[r]un_queue.py'
