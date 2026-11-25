set -eu
date -Is
free -h
for p in /data1/shenglin.zeng /data5/shenglin.zeng;do
 if test -d "$p";then
  df -h "$p"
  find "$p" -maxdepth 4 -type d \( -name '*sdxl*' -o -name '*10k*' -o -name 'env' -o -name 'envs' \) | head -25
 fi
done
nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits
