set -euo pipefail
hostname
uname -m
df -h /data1/shenglin.zeng
free -h
nvidia-smi --query-gpu=index,uuid,pci.bus_id,driver_version,name,memory.free,utilization.gpu --format=csv
ls -ld /data1/shenglin.zeng /data1/shenglin.zeng/*
for p in /data1/shenglin.zeng/miniconda3/envs/sherly/bin/python /data1/shenglin.zeng/miniconda/envs/sherly/bin/python; do
 if test -x "$p"; then CUDA_VISIBLE_DEVICES='' "$p" -c 'import sys,importlib.util; print(sys.executable,sys.version); print({m:bool(importlib.util.find_spec(m)) for m in ["torch","diffusers","transformers","cleanfid","conda_pack"]})'; fi
done
