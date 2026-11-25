free -h
nvidia-smi -i 4 --query-gpu=index,uuid,memory.free,utilization.gpu --format=csv,noheader,nounits
find /data1/shenglin.zeng/coco_missing_20261111 -name '*.partial' -printf '%s %p\n' | head -8
