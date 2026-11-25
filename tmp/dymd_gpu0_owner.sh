nvidia-smi -i 0 --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
nvidia-smi -i 0
ps -p 2026 -o user,pid,comm,etime,%cpu
