cd /data5/shenglin.zeng/sdxl_crossdistill
date -Is
for f in outputs/research_small_20261022/*.exit; do echo "$f $(cat "$f")"; done
for pid in 1877900 1877901 1884787 1884788 1888491 1888492 1897610 1897611; do
 if test -d "/proc/$pid"; then echo "PID_STILL_PRESENT $pid"; else echo "OUR_WORKER_EXITED $pid"; fi
done
nvidia-smi -i 0,3 --query-compute-apps=gpu_uuid,pid,used_memory --format=csv
du -sh outputs/research_small_20261022
