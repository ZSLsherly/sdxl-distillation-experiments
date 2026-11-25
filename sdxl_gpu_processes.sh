nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv
ps -eo pid,user,args | grep -E '[s]dxl_(train|compare|prepare|download|export|summarize)' | head -n 15
