hostname
id
pwd
nvidia-smi --query-gpu=index,name,driver_version,memory.free --format=csv
df -h / /data /data1 /data2 /home 2>/dev/null
ls -d /data* /opt/*conda* /home/* /root/*conda* /mnt/* 2>/dev/null
find /data /data1 /data2 /home /root -maxdepth 4 -type d \( -iname '*wan*' -o -iname '*physis*' -o -name 'envs' -o -name '*codex*' \) 2>/dev/null | head -80
command -v python python3 conda git
free -h
