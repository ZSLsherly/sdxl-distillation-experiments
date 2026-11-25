id
nvidia-smi -i 4,5,6,7 --query-gpu=index,name,memory.free,utilization.gpu,pstate --format=csv
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv
nvidia-smi topo -m
df -h /data1
ls /data1/shenglin.zeng/miniconda3/envs
cat /home/shenglin.zeng/.conda/environments.txt 2>/dev/null
command -v rsync sshpass
python3 -c 'import paramiko; print("system paramiko",paramiko.__version__)'
find /data1/shenglin.zeng/miniconda3/envs -maxdepth 4 -path '*/site-packages/torch' -type d
