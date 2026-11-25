source /data5/shenglin.zeng/env.sh
uptime
free -h
vmstat 1 2
nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv
tail -n 3 /data5/shenglin.zeng/sdxl_crossdistill/logs/pilot_assessment.log
