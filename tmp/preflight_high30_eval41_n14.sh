set -eu
date -Iseconds
free -h
df -h /data1 /data2 2>/dev/null || true
ls -d /data*/shenglin.zeng/* 2>/dev/null
for p in /data1/shenglin.zeng/sdxl_eval_uniform_20261025 /data1/shenglin.zeng/sdxl_eval /data1/shenglin.zeng; do
 test ! -d "$p" || find "$p" -maxdepth 4 \( -name pyvenv.cfg -o -name manifest.json -o -name inception-2015-12-05.pt -o -name revision.txt -o -name rclone \) -print | head -45
done
nvidia-smi -i 1,2,3,5,6,7 --query-gpu=index,uuid,memory.free,utilization.gpu --format=csv,noheader,nounits
nvidia-smi -i GPU-dee80cca-d29a-9086-d006-b76ba6ee093e --query-compute-apps=pid --format=csv,noheader,nounits; echo GPU_QUERY_RC=$?
